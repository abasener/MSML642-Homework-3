#!/usr/bin/env python3
"""Optional perception -> state -> planning -> execution ROS 2 skeleton.

This file wires up the sensors and the course-provided waypoint Action. It
does not contain a map, gate coordinates, gate width, route, or working policy.
Students may replace the entire file with their own ROS 2 node.
"""

import time
from enum import Enum, auto

# ADDED:
import math
from dataclasses import dataclass
import numpy as np
import colorsys
import os
from collections import deque


import rclpy
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from hw3_waypoint.action import DriveToPose


# Testing?
DEBUG_MAP = True
TEST_BLOCKED = False # run a test to go stright and get blocked
#set up
GOAL_XY = (0.0, 4.4)
MAX_SENSOR_AGE_S = 2.0
# Tunable parms
ROBOT_RADIUS = 0.25
SAFETY_MARGIN = 0.05
MAX_STEP = 0.5
MIN_STEP = 0.10
MAX_FAILS = 4
MAX_LEG = 1.0
RL_RADIUS = 0.5
W_SIZE = 0.1
W_DIST = 1.0
W_GOAL = 0.3
# gate parms
GATE_MIN_EDGE = 4
GATE_MIN_GAP = 0.45
GATE_MAX_GAP = 1.6
GATE_CONFIRM = 2
APPROACH_D = 0.6
PASS_D = 0.6
APPROACH_TOL = 0.2
ALIGN_TOL = 0.35
ESCAPE_MIN = 0.12 
HYST = 3.0
KEEP_R = 0.3
GATE_MERGE = 2
GATE_FREE_FRAC = 0.6
W_GREEN = 4.0



class Phase(Enum):
    OBSERVE = auto()
    WAIT_FOR_MOTION = auto()
    DONE = auto()
    FAILED = auto()


# added classes, functoins and objects outside of PolicyExample

class Planner:
    # tunable parms:
    MIN_FRONTIER = 5

    def __init__(self, grid, clearance):
        self.grid = grid
        # add padding clearance to walls
        self.clearCells = int(math.ceil(clearance / grid.res))

        r = self.clearCells
        self.diskOS = [(dr,dc)
                       for dr in range(-r, r+1)
                       for dc in range(-r, r+1)
                       if dr *dr + dc * dc <= r*r]
        self.hard = None
        self.blocked = None
        self.dist = None
        self.parent = None

    def biuldBlocked(self, extra=None):
        g = self.grid
        lab = g.lables
        solid = (lab == g.WALL) | (lab == g.RED_EDGE) | (lab == g.GREEN_EDGE)
        if extra is not None:
            solid = solid | extra
        unknown = lab == g.UNKNOWN
        
        # Inflate to fine edges
        R,C = solid.shape
        inflated = np.zeros_like(solid)
        for dr, dc in self.diskOS:
            inflated[max(dr,0):R + min(dr,0), max(dc,0):C + min(dc,0)] |= \
            solid[max(-dr,0):R + min(-dr,0), max(-dc,0):C + min(-dc,0)]

        self.hard = solid | unknown
        self.blocked = inflated | unknown
        return self.blocked

    def flooded(self, start):
        R,C = self.blocked.shape
        sr,sc = start
        self.dist = np.full((R,C), -1, dtype=np.int32)
        self.parent = np.full(R*C, -1, dtype=np.int32)
        if not self.grid.inBound(sr,sc) or self.blocked[sr,sc]:
            return False

        self.dist[sr,sc]=0
        q = deque([(sr,sc)])
        blocked, dist, parent = self.blocked, self.dist, self.parent

        # get the oldest cell spred out
        while q:
            r,c = q.popleft()
            nd = dist[r,c] +1
            for nr, nc in ((r+1,c), (r-1,c), (r,c+1),(r,c-1)):
                if (0 <= nr < R and 0 <= nc < C
                    and dist[nr, nc] < 0 and not blocked[nr,nc]):
                    dist[nr,nc] = nd
                    parent[nr*C+nc] = r*C+c
                    q.append((nr,nc))
        return True

    def escapeCell(self,start,maxCells=400):
        R,C= self.blocked.shape
        seen = {start}
        q = deque([start])
        while q and len(seen) < maxCells:
            r,c = q.popleft()
            if not self.blocked[r,c]:
                return (r,c)
            for nr, nc in ((r+1,c), (r-1,c),(r,c+1), (r,c-1)):
                if (0 <= nr < R and 0 <= nc < C and (nr,nc) not in seen
                        and not self.hard[nr,nc]):
                    seen.add((nr,nc))
                    q.append((nr,nc))
        return None

    def path2(self,target):
        tr,tc = target
        if self.dist is None or self.dist[tr,tc] < 0:
            return None
        C = self.blocked.shape[1]
        path=[]
        idx = tr*C+tc
        while idx >= 0:
            path.append(divmod(idx,C))
            idx = self.parent[idx]
        path.reverse()
        return path

    def lineClear(self,a,b):
        for r,c in occupancyGrid.lineCells(a[0], a[1], b[0], b[1]):
            if self.blocked[r,c]:
                return False
        return True

    def simplify(self,path):
        if not path or len(path) < 2:
            return []
        wps = []
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.lineClear(path[i], path[j]):
                j -= 1
            wps.append(path[j])
            i = j
        return wps

    def frontiers(self):
        g = self.grid
        lab = g.lables
        free = lab == g.FREE
        unk = lab == g.UNKNOWN

        nb = np.zeros_like(unk)
        nb[1:, :] |= unk[:-1,:]
        nb[:-1, :] |= unk[1:, :]
        nb[:, 1:] |= unk[:, :-1]
        nb[:, :-1] |= unk[:, 1:]
        front = free & nb

        R,C = front.shape
        seen = np.zeros_like(front)
        cluster = []
        for r0, c0 in zip(*np.nonzero(front)):
            if seen[r0,c0]:
                continue
            seen[r0,c0] = True
            cells, q = [], deque([(r0,c0)])
            while q:
                r,c = q.popleft()
                cells.append((r,c))
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        nr, nc = r + dr, c+ dc
                        if(0 <= nr < R and 0 <= nc <C
                           and front[nr,nc] and not seen[nr,nc]):
                            seen[nr,nc] = True
                            q.append((nr,nc))
            if len(cells) >= self.MIN_FRONTIER:
                cluster.append(cells)
        return cluster

    def nearestReachable(self, cells):
        best = None
        for r,c in cells:
            d = self.dist[r,c]
            if d >= 0 and (best is None or d < best[1]):
                best = ((r,c),d)
        return best


class occupancyGrid:
    # what the sesors see saved as a grid
    # [row, col] = [y, x]
    
    # Parms adjust these as needed
    LIDARosX, LIDARosY = -0.064, 0.0    # the off set of the LiDAR readings to the odem readings
    camOSx,camOSy = 0.076, 0.0          # camera mounting offset 
    InfFreeRang = 3.0
    DEBUG_DIR = "/workspace/Part2/debug" # output folder for debuging logs/images
        # Grid sizing set up
    res = 0.05  # meters per cell
    Xmin, Xmax = -6.0, 6.0
    Ymin, Ymax = -6.0, 6.0
        # min/max for lidar score
    scoreMin, scoreMax = -10.0, 10.0
        # Amount to adust when see evidance for/agenst wall
    hitEv = -3.0
    freeEv = 1.0
        # threshholds for if lidar score is a wall or free
    wallThr = -3.0
    freeThr = 2.0
        # Camara color thresh
    greenMargR,greenMargB = 30,20
    redMargG, RedMargB = 30,20
    yelMargB, yelMaxRG = 60, 50
    horBand = 20                    # pixles above/bellow lidare scan plan
    colorFracThr = 0.5              # fractoin of band px of the color to count
    maxCamLag = 0.1                 # the max the lag/missmatch between lidar and camra
        # bit flags
    flagG = 1<<1
    FLAGr = 1<<2
    


    UNKNOWN, FREE, WALL, RED_EDGE, GREEN_EDGE = 0, 1, 2, 3, 4
    CamSeen = 1 << 0

    def __init__(self):
        self.originX = self.Xmin
        self.originY = self.Ymin
        self.cols = int(round((self.Xmax - self.Xmin) / self.res))
        self.rows = int(round((self.Ymax - self.Ymin) / self.res))
        shape = (self.rows, self.cols)

        self.scores = np.full(shape, np.nan, dtype=np.float32)      # Lidar score starts at nan for unseen
        self.lables = np.full(shape, self.UNKNOWN, dtype=np.uint8)  # Simple lables for what is in the map
        self.flags = np.zeros(shape, dtype=np.uint8)                # 
        self.traveled = np.zeros(shape, dtype=np.int32)             # Tracks where the robot has moved using step
        self.step = 0                                               # counds up each space moved though
        self._LastScanStamp = None
        self._LastCell = None
        self.edgeGuard = 8


    def world2grid(self, x,y):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"world to grid got invalid x or y: ({x},{y})")
        col = math.floor((x - self.originX) / self.res)
        row = math.floor((y - self.originY) / self.res)
        return row, col
    
    def grid2world(self, row, col):
        x = self.originX + (col + 0.5) * self.res
        y = self.originY + (row + 0.5) * self.res
        return x,y

    def inBound(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def worldInBounds(self, x,y):
        return self.inBound(*self.world2grid(x,y))

    def addEv(self, r, c, value):
        s = self.scores[r,c]
        s = value if math.isnan(s) else s + value
        self.scores[r,c] = min(self.scoreMax, max(self.scoreMin, s))

    def updateFromLidar(self, obs):
        if obs.scanStamp == self._LastScanStamp:
            return False
        self._LastScanStamp = obs.scanStamp

        # get Lidar pos in map
        cy, sy = math.cos(obs.yaw), math.sin(obs.yaw)
        lx = obs.x + self.LIDARosX * cy - self.LIDARosY * sy
        ly = obs.y + self.LIDARosX * sy + self.LIDARosY * cy
        r0, c0 = self.world2grid(lx, ly)
        if not self.inBound(r0, c0):
            return False

        hits = []
        for i, d in enumerate(obs.ranges):
            if math.isnan(d) or d < obs.rangeMin:
                continue
            hit = math.isfinite(d) and d <= obs.rangeMax
            length = d if hit else self.InfFreeRang
            a = obs.yaw + obs.angleMin + i * obs.angleInt
            r1, c1 = self.world2grid(lx + length * math.cos(a),
                                     ly + length * math.sin(a))
            cells = self.lineCells(r0,c0, r1,c1)
            for r,c in (cells[:-1] if hit else cells):
                if not self.inBound(r,c):
                    break
                self.addEv(r,c, self.freeEv)
            if hit and self.inBound(r1,c1):
                hits.append((r1,c1))

        for r , c in hits:
            self.addEv(r, c, self.hitEv)
        return True

    def updateFromCamera(self,obs):
        rgb = self.image2array(obs.image)
        if rgb is None:
            return []
    
        if abs(obs.imageStamp - obs.scanStamp) > self.maxCamLag:
            return []      # img and lidar not from same time
        
        green, red, yellow = self.colorMasks(rgb)
        h,w = green.shape

        # rows to sample
        r0 = max(0, int(obs.cy) - self.horBand)
        r1 = min(h, int(obs.cy) + self.horBand + 1)

        # mean over rows of a bool col is fract of true px
        gCol = green[r0:r1,:].mean(axis=0)
        rCol = red[r0:r1,:].mean(axis=0)

        # make sure col only counts if every col next to it within edgeGuard are also that color
        gOk = gCol >= self.colorFracThr
        rOk = rCol >= self.colorFracThr
        gSafe, rSafe = gOk.copy(), rOk.copy()
        for s in range(1, self.edgeGuard + 1):
            gSafe[s:] &= gOk[:-s]      # left
            gSafe[:-s] &= gOk[s:]      # right
            rSafe[s:] &= rOk[:-s]
            rSafe[:-s] &= rOk[s:]

        cyaw, syaw = math.cos(obs.yaw), math.sin(obs.yaw)
        # Lidar positoin in robot
        lxr,lyr = self.LIDARosX, self.LIDARosY
        #lidar positoin in map
        lx = obs.x + lxr * cyaw - lyr * syaw
        ly = obs.y + lxr * syaw + lyr * cyaw
        rS, cS = self.world2grid(lx, ly)

        hints = []
        # loop each lidar beem fro where it hits in the image
        for i, d in enumerate(obs.ranges):
            if math.isnan(d) or d < obs.rangeMin:
                continue
            hit = math.isfinite(d) and d <= obs.rangeMax
            length = d if hit else self.InfFreeRang     # inf free rang
            a = obs.angleMin + i * obs.angleInt         # Beam angle

            # end point in the robot frame
            px = lxr + length * math.cos(a)
            py = lyr + length * math.sin(a)

            # get same point relative to camra
            xc = px - self.camOSx
            yc = py - self.camOSy
            if xc < 0.05:
                continue    # behind cam
            
            u = int(round(obs.cx - obs.fx * yc / xc))
            if not (0<= u < w):
                continue    # outside the cam fov

            # end point on map 
            mx = obs.x + px * cyaw - py * syaw
            my = obs.y + px * syaw + py * cyaw
            rE,cE = self.world2grid(mx,my)

            # Cam seen every cell alone lidar beem in cam POV
            for r,c in self.lineCells(rS, cS, rE, cE):
                if not self.inBound(r,c):
                    break
                self.flags[r,c] |= self.CamSeen

            # what is color in col at lidar hight
            if gSafe[u]:
                color = "green"
            elif rSafe[u]:
                color = "red"
            else:
                continue

            if hit and self.inBound(rE, cE):
                if color == "green":
                    self.flags[rE,cE] |= self.flagG
                    self.flags[rE,cE] &= ~self.FLAGr & 0xFF
                else:
                    self.flags[rE,cE] |= self.FLAGr
                    self.flags[rE,cE] &= ~self.flagG & 0xFF
            elif not hit:
                bearing = obs.yaw + a
                hints.append((obs.x,obs.y,bearing,color))
        return hints 
            




        

    def updateLables(self):
        s = self.scores
        new = np.full(s.shape, self.UNKNOWN, dtype=np.uint8)
        with np.errstate(invalid="ignore"):
            new[s >= self.freeThr] = self.FREE
            new[s <= self.wallThr] = self.WALL

        isWall = new == self.WALL
        new[isWall & ((self.flags & self.flagG) != 0)] = self.GREEN_EDGE
        new[isWall & ((self.flags & self.FLAGr) != 0)] = self.RED_EDGE
        
        self.lables = new

    def recorPose(self, x, y):
        r,c = self.world2grid(x,y)
        if not self.inBound(r,c) or (r,c) == self._LastCell:
            return
        self.step += 1
        if self._LastCell is None:
            cells = [(r,c)]
        else:
            cells = self.lineCells(*self._LastCell, r,c)
        for rr, cc in cells:
            self.traveled[rr,cc] = self.step
        self._LastCell = (r,c)

    def saveDebugImg(self, filename, goal = None):
        colors = {self.UNKNOWN: (128,128,128), self.FREE: (255,255,255),
                  self.WALL: (0,0,0), self.RED_EDGE:(255,0,0),
                  self.GREEN_EDGE: (0,200,0)}
        img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        for label, rgb in colors.items():
            img[self.lables == label] = rgb
        # gates red/green
        rs,cs = np.nonzero(self.traveled)
        if len(rs):
            t = self.traveled[rs,cs]
            lo, hi = t.min(), t.max()
            for r,c,v in zip(rs,cs,t):
                frac = (v - lo) / max(1,hi-lo)
                rgb = colorsys.hsv_to_rgb(0.67 * (1 - frac), 1.0, 1.0)
                img[r,c] = [int(255*k) for k in rgb]
        # goal yellow
        if goal is not None:
            gx, gy = goal
            ra, ca = self.world2grid(gx - 0.5, gy - 0.5)
            rb, cb = self.world2grid(gx + 0.5, gy + 0.5)
            for r in range(ra, rb +1):
                for c in range(ca, cb +1):
                    edge = r in (ra, rb) or c in (ca,cb)
                    if edge and self.inBound(r,c):
                        img[r,c] = (255, 220, 0)

        img = np.flipud(img)
        os.makedirs(self.DEBUG_DIR, exist_ok=True)
        with open(os.path.join(self.DEBUG_DIR, filename), "wb") as f:
            f.write(f"P6 {self.cols} {self.rows} 255\n".encode())
            f.write(img.tobytes())

    def saveDecisionImg(self, filename, decisions, goal=None):
        colors = {self.UNKNOWN: (128,128,128), self.FREE: (255,255,255),
                        self.WALL:(0,0,0), self.RED_EDGE:(255,0,0),
                        self.GREEN_EDGE:(0,200,0)}
        img = np.zeros((self.rows, self.cols,3), dtype=np.uint8)
        for label, rgb in colors.items():
            img[self.lables == label] =rgb
        img[self.traveled > 0] = (190,190,190)

        if goal is not None:
            gx, gy = goal
            ra, ca = self.world2grid(gx - 0.5, gy - 0.5)
            rb, cb = self.world2grid(gx + 0.5, gy + 0.5)
            for r in range(ra, rb+1):
                for c in range(ca,cb+1):
                    if (r in (ra,rb) or c in (ca,cb)) and self.inBound(r,c):
                        img[r,c] = (255,255,0)
        #marker shape
        xShp = [(d,d) for d in range(-2,3)] + [(d,-d) for d in range(-2,3)]
        oShp = [(dr,dc) for dr in range(-3,4) for dc in range(-3,4) if 2.5 <= math.hypot(dr,dc) <= 3.5]

        n = len(decisions)
        for k, (x,y,kind) in enumerate(decisions):
            frac = k/ max(1,n-1)
            rgb = [int(255*v) for v in colorsys.hsv_to_rgb(0.67 * (1-frac), 1.0, 1.0)]
            r0, c0 = self.world2grid(x,y)
            for dr, dc in (xShp if kind == "W" else oShp):
                r,c = r0 + dr, c0 + dc
                if self.inBound(r,c):
                    img[r,c] = rgb

        self.writePPM(filename, np.flipud(img))

    def saveSnapshot(self, filename, pose):
        os.makedirs(self.DEBUG_DIR, exist_ok=True)
        np.savez(os.path.join(self.DEBUG_DIR, filename), lables=self.lables, pose=np.array(pose))


    def freeDis(self, x,y, heading, maxDist, halfWidth):
        step = self.res/2
        fx,fy = math.cos(heading), math.sin(heading)
        px, py = -fy, fx

        offsets = np.arange(-halfWidth, halfWidth + 1e-9, self.res)

        d = 0.0
        # go forward along heding half cell at a time
        while d + step <= maxDist:
            nd = d +step
            cx = x + nd * fx
            cy = y + nd * fy
            # check allong path for ishue
            for off in offsets:
                r,c = self.world2grid(cx + off *px, cy + off * py)
                if not self.inBound(r,c) or self.lables[r,c] != self.FREE:
                    return d
            d = nd
        return d

    def colorMasks(self,rgb):
        r = rgb[:,:,0].astype(np.int16)
        g = rgb[:,:,1].astype(np.int16)
        b = rgb[:,:,2].astype(np.int16)

        green = (g > r + self.greenMargR) & (g > b + self.greenMargB)
        red = (r > g + self.redMargG) & (r > b + self.RedMargB)
        yellow = ((r>b + self.yelMargB) & (g>b+self.yelMargB) & (np.abs(r-g) < self.yelMaxRG))
        return green, red, yellow

    def writePPM(self, filename, img):
        h,w = img.shape[:2] # get width, hight of image
        os.makedirs(self.DEBUG_DIR, exist_ok=True)
        with open(os.path.join(self.DEBUG_DIR, filename), "wb") as f:
            f.write(f"P6 {w} {h} 255\n".encode())
            f.write(np.ascontiguousarray(img).tobytes())

    def saveCameraDebug(self, obs):
        rgb = self.image2array(obs.image)
        if rgb is None:
            return None
        green, red, yellow = self.colorMasks(rgb)

        self.writePPM("camra.ppm", rgb)

        mask = rgb //4
        mask[green] = (0, 255, 0)
        mask[red] = (255,0,0)
        mask[yellow] = (255,220,0)
        self.writePPM("mask.ppm", mask)

        return green.mean(), red.mean()

    @staticmethod
    def image2array(img):
        enc = img.encoding.lower()
        if enc not in ("rgb8", "bgr8"):
            return None
        buf = np.frombuffer(img.data, dtype=np.uint8) # get raw bytes as numbers
        rows = buf.reshape(img.height, img.step) 
        arr = rows[:,:img.width*3].reshape(img.height, img.width, 3) # keep only real pixels and split rows
        if enc == "bgr8": # check if chanels need reversing
            arr = arr[:,:,::-1]
        return arr
    
    @staticmethod
    def lineCells(r0,c0,r1,c1):
        # get list of cells from r0,c0 to r1,c1 in order for Lidar beem use
        cells = []
        dr, dc = abs(r1 - r0), abs(c1-c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dc - dr
        r,c = r0,c0
        # loop though to get cells
        while True:
            cells.append((r,c))
            if r == r1 and c == c1:
                break
            e2 = 2 *err
            if e2 > -dr:    # step to next col
                err -= dr
                c += sc
            if e2 < dc:     # step to next row
                err += dc
                r += sr
        return cells

@dataclass
class Gate:
    # inner edge cells of sides
    a:tuple
    b:tuple
    # gap ceneter
    cx:float
    cy:float
    # perpenducular to gap
    nx:float
    ny:float
    # line crossing the gap of the gate
    width:float
    lineMask:object

def findClusters(mask, minSize):
    R,C =mask.shape
    seen = np.zeros_like(mask)
    clusters = []
    for r0,c0 in zip(*np.nonzero(mask)):
        if seen[r0,c0]:
            continue
        seen[r0,c0] = True
        cells,q = [], deque([(r0, c0)])
        while q:
            r,c = q.popleft()
            cells.append((r,c))
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C and mask[nr,nc] and not seen[nr, nc]:
                        seen[nr,nc] = True
                        q.append((nr,nc))
        if len(cells) >= minSize:
            clusters.append(cells)
    return clusters

class GateTracker:
    def __init__(self,grid):
        self.grid = grid
        self.count = {"green":0, "red":0}
        self.last = {"green":None, "red":None}
        self.why = {"green": "", "red": ""}

    def detect(self,color):
        g = self.grid
        label = g.GREEN_EDGE if color == "green" else g.RED_EDGE
        raw = g.lables == label

        # merg close color so close ones become one wall
        R, C = raw.shape
        k = GATE_MERGE
        grown = raw.copy()
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                grown[max(dr, 0):R + min(dr, 0), max(dc, 0):C + min(dc, 0)] |= \
                    raw[max(-dr, 0):R + min(-dr, 0), max(-dc, 0):C + min(-dc, 0)]
        clusters = []
        for cells in findClusters(grown, 1):
            real = [(r, c) for r, c in cells if raw[r, c]]
            if len(real) >= GATE_MIN_EDGE:
                clusters.append(real)
        if len(clusters) < 2:
            self.why[color] = f"{len(clusters)} edge cluster(s) so far"
            return None
        clusters.sort(key=len, reverse=True)

        # get best valid pairs of the color
        best = None
        top = clusters[:4]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                A, B = np.array(top[i]), np.array(top[j])
                d = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
                ii, jj = np.unravel_index(np.argmin(d), d.shape)
                width = float(d[ii, jj]) * g.res
                if not (GATE_MIN_GAP <= width <= GATE_MAX_GAP):
                    continue
                a = tuple(int(v) for v in A[ii])
                b = tuple(int(v) for v in B[jj])
                between = occupancyGrid.lineCells(a[0], a[1], b[0], b[1])[1:-1]
                if not between:
                    continue
                freeFrac = sum(g.lables[r, c] == g.FREE for r, c in between) / len(between)
                if freeFrac < GATE_FREE_FRAC:
                    continue
                size = len(top[i]) + len(top[j])
                if best is None or size > best[0]:
                    best = (size, a, b, width)
        if best is None:
            self.why[color] = f"no valid pair; cluster sizes {[len(c) for c in top]}"
            return None

        _, a, b, width = best
        self.why[color] = f"ok: width {width:.2f} m"
        ax, ay = g.grid2world(*a)
        bx, by = g.grid2world(*b)
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        ux, uy = (bx - ax) / width, (by - ay) / width
        nx, ny = -uy, ux
        mask = np.zeros(g.lables.shape, dtype=bool)
        for r, c in occupancyGrid.lineCells(a[0], a[1], b[0], b[1]):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if g.inBound(r + dr, c + dc):
                        mask[r + dr, c + dc] = True
        return Gate(a, b, cx, cy, nx, ny, width, mask)

    def update(self,color):
        gate = self.detect(color)
        prev = self.last[color]
        if gate is None:
            # keep past gates and count
            return prev, prev is not None and self.count[color] >= GATE_CONFIRM
        if prev is not None and math.hypot(gate.cx - prev.cx, gate.cy - prev.cy) < 0.3:
            self.count[color] += 1
        else:
            self.count[color] = 1
        self.last[color] = gate
        return gate, self.count[color] >= GATE_CONFIRM


@dataclass
class Observation:
    x: float            # metres, map frame
    y: float
    yaw: float          # radians
    ranges: tuple       # scan range
    angleMin: float
    angleInt: float
    rangeMin: float
    rangeMax: float
    scanStamp: float
    image: object
    fx: float           # camra focal length
    cx: float           # camra center picel col
    cy: float           # camra center in Row
    imageStamp: float

    
    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw)
    
class PolicyExample(Node):
    def __init__(self):
        super().__init__('policy_example')
        self.phase = Phase.OBSERVE
        self.odom = self.scan = self.image = self.camera_info = None
        self.received_at = {'odom': 0.0, 'scan': 0.0, 'image': 0.0}
        self.observe_after = 0.0
        self.belief = None  # Persistent world state, built only from observations.
        self.last_waypoint = None
        self.last_result_code = None
        self.action = ActionClient(self, DriveToPose, '/drive_to_pose')
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_camera_info,
                                 qos_profile_sensor_data)
        self.create_timer(0.2, self.tick)

        # Added
        self.grid = occupancyGrid()
        self.planner = Planner(self.grid, ROBOT_RADIUS + SAFETY_MARGIN)
        self.guideCell = None   # the (row, col) of the giuld the robot is moving to
        self.blacklist = []     # keep track of black listed areas (fully explored / deadend)
        self.decisionLog = []   # (x,y.kind) 
        self.gates = GateTracker(self.grid)
        self.stage = "FIND_GATE"
        self.gate = None
        self.nearSign = 1.0
        self.crossStep = "APPROACH"
        self.redMask = None
        self.greenMask = None
        self.startXY = None 
        self.get_logger().info(
            f"grid {self.grid.rows}x{self.grid.cols} cells at {self.grid.res} m"
        )
        self._last_debug_save = 0.0

        self.failCount = 0
        self._testBlockedSent = False
        self.colorHints = []

    def on_odom(self, msg):
        self.odom = msg
        self.received_at['odom'] = time.monotonic()

    def on_scan(self, msg):
        self.scan = msg
        self.received_at['scan'] = time.monotonic()

    def on_image(self, msg):
        self.image = msg
        self.received_at['image'] = time.monotonic()

    def on_camera_info(self, msg):
        self.camera_info = msg

    def observations_ready(self):
        now = time.monotonic()
        if any(value is None for value in
               (self.odom, self.scan, self.image, self.camera_info)):
            return False
        if self.odom.header.frame_id != 'map':
            return False
        return all(self.observe_after <= received <= now and
                   now - received <= MAX_SENSOR_AGE_S
                   for received in self.received_at.values())

    def turn2Open(self, obs):
        clear = ROBOT_RADIUS +SAFETY_MARGIN
        look = MAX_STEP + clear

        left = self.grid.freeDis(obs.x, obs.y, obs.yaw + math.pi/2, look, clear)
        right = self.grid.freeDis(obs.x, obs.y, obs.yaw - math.pi/2, look, clear)

        if max(left, right) - clear < MIN_STEP:
            turn = math.pi
        elif left >= right:
            turn = math.pi/2
        else:
            turn = -math.pi/2
        
        yawN = math.atan2(math.sin(obs.yaw + turn), math.cos(obs.yaw +turn))
        self.get_logger().info(
            f"TURN: left={left:.2f} right={right:.2f} -> {math.degrees(turn):+.0f}deg"
        )
        return (obs.x, obs.y, yawN)

    def perceive(self, odom, scan, image, camera_info):
        """TODO: Return observations derived from these live ROS messages.

        Use odometry for pose, LiDAR for observed free/occupied space, and the
        camera for color evidence. Associate camera color with LiDAR geometry
        using CameraInfo and the robot's sensor mounting; estimate any
        opening's location/width
        from measurements. A missing or occluded gate stays unknown. Do not
        read the SDF world or insert gate coordinates from either sample map.

        Return None until implemented; None means no motion will be sent.
        """
        log = self.get_logger()
        T = 2.0  # limt timer print per sec

        # odometry
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                        1 - 2 * (q.y * q.y + q.z * q.z))
        log.info(f"ODOM x={p.x:.2f} y={p.y:.2f} yaw={math.degrees(yaw):.1f}deg "
                f"frame={odom.header.frame_id}", throttle_duration_sec=T)

        # Lidar
        r = scan.ranges
        n_inf = sum(1 for d in r if math.isinf(d))
        n_nan = sum(1 for d in r if math.isnan(d))
        i_front = round((0.0 - scan.angle_min) / scan.angle_increment)
        log.info(f"SCAN n={len(r)} angle_min={math.degrees(scan.angle_min):.1f} "
                f"angle_max={math.degrees(scan.angle_max):.1f} "
                f"inc={math.degrees(scan.angle_increment):.2f}deg "
                f"inf={n_inf} nan={n_nan} front[i={i_front}]={r[i_front % len(r)]:.2f}",
                throttle_duration_sec=T)

        # camara
        off = (image.height // 2) * image.step + 3 * (image.width // 2)
        rgb = tuple(image.data[off:off + 3])
        log.info(f"IMAGE {image.width}x{image.height} enc={image.encoding} "
                f"step={image.step} bytes={len(image.data)} center_rgb={rgb} "
                f"fx={camera_info.k[0]:.1f} cx={camera_info.k[2]:.1f}",
                throttle_duration_sec=T)

        stamp = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        imgStamp = image.header.stamp.sec + image.header.stamp.nanosec * 1e-9


        return Observation(x=p.x, y=p.y, yaw=yaw,imageStamp=imgStamp,
                    cy=camera_info.k[5],
                    image = image, fx=camera_info.k[0], cx=camera_info.k[2],
                    ranges=scan.ranges, angleMin=scan.angle_min,
                    angleInt=scan.angle_increment,
                    rangeMin=scan.range_min, rangeMax=scan.range_max,
                    scanStamp=stamp)

    def update_state(self, previous_belief, perception):
        """TODO: Fuse observations into a persistent, uncertain world state.

        Distinguish UNKNOWN from observed FREE and OCCUPIED cells. Track the
        robot pose, candidate openings and their color/width confidence, and
        which places have been explored. Never turn unseen space into free
        space merely because the planner would prefer a shorter path.
        """

        # return true for for unseen scan
        if self.grid.updateFromLidar(perception):
            self.colorHints = self.grid.updateFromCamera(perception)
        self.grid.updateLables()
        # where the robot has been
        self.grid.recorPose(perception.x, perception.y)

        # check were next wall is vs expected
        if not getattr(self, "_wall_check_done", False):
            r0, c0 = self.grid.world2grid(perception.x, perception.y)
            for r in range(r0, self.grid.rows):
                if self.grid.lables[r, c0] == occupancyGrid.WALL:
                    wx, wy = self.grid.grid2world(r, c0)
                    self.get_logger().info(f"WALL CHECK: first wall ahead at y={wy:.3f}")
                    break
            self._wall_check_done = True

        # update at most every 2 seconds
        now = time.monotonic()
        if DEBUG_MAP and now - self._last_debug_save > 2.0:
            self.grid.saveDebugImg("map_3.ppm", goal=GOAL_XY)
            self.grid.saveDecisionImg("decisions_3.ppm", self.decisionLog, goal=GOAL_XY)
            frac = self.grid.saveCameraDebug(perception)
            self.grid.saveSnapshot("snap.npz", (perception.x, perception.y, perception.yaw))
            if frac is not None:
                self.get_logger().info(
                    f"Color: green={frac[0]:.3f} red={frac[1]:.3f}")
            self._last_debug_save = now 

        self.get_logger().info(
            f"STAMPS image-scan = {perception.imageStamp - perception.scanStamp:+.3f}s",
            throttle_duration_sec=2.0)
                
        return perception

    def goal_reached(self, belief):
        """TODO: Check the observed robot pose against GOAL_XY."""
        gx, gy = GOAL_XY
        fx = belief.x - 0.064 * math.cos(belief.yaw)
        fy = belief.y - 0.064 * math.sin(belief.yaw)
        reach = 0.5 + 0.10
        return abs(fx - gx) <= reach and abs(fy - gy) <= reach

    def plan_next_waypoint(self, belief, last_waypoint, last_result_code):
        """TODO: Return a safe map-frame (x, y, theta), or None.

        If the green opening has not been observed and measured, choose an
        exploration/observation waypoint in known free space. Once it has,
        plan through that opening only if its measured clearance fits the
        robot plus a safety margin. Never cross a red opening. Replan from
        updated observations after every Action result, especially BLOCKED;
        do not resend an unchanged blocked waypoint.

        Return None when no waypoint is currently justified by the belief.
        You may use a state machine, a different planner, or a learned policy.
        """
        obs = belief
        g, p = self.grid, self.planner
        if self.startXY is None:
            self.startXY = (obs.x, obs.y)
            self.saveStage(0)

        # test hook drive into a wall for testing
        if TEST_BLOCKED and not self._testBlockedSent:
            self._testBlockedSent = True
            d = 3.0
            self.get_logger().warn("Test blocked -> Driving into wall")
            return (obs.x +d * math.cos(obs.yaw), obs.y + d * math.sin(obs.yaw), obs.yaw)

        # failed move: dont try again soon
        if last_result_code in ("BLOCKED", "STALLED"):
            if self.stage == "CROSS_GATE":
                self.crossStep = "APPROACH"
            elif self.guideCell is not None:
                self.blacklist.append(self.guideCell)
                self.get_logger().info(f"BLACKLIST {self.guideCell} after {last_result_code}")

        # gates from curent known map
        red, _ = self.gates.update("red")
        if red is not None:
            self.redMask = red.lineMask if self.redMask is None else (self.redMask | red.lineMask)
        green, greenOk = self.gates.update("green")

        if self.stage == "FIND_GATE":
            self.get_logger().info(
                f"GATE? green count={self.gates.count['green']} ({self.gates.why['green']}) | "
                f"red: {self.gates.why['red']}", throttle_duration_sec=4.0)
            
        if self.stage == "FIND_GATE" and greenOk:
            self.gate = green
            sx, sy = self.startXY
            side = (sx - green.cx) * green.nx + (sy - green.cy) * green.ny
            self.nearSign = 1.0 if side >= 0 else -1.0
            self.stage, self.crossStep = "CROSS_GATE", "APPROACH"
            self.get_logger().info(
                f"GATE green confirmed: centre ({green.cx:.2f},{green.cy:.2f}) "
                f"width {green.width:.2f} m -> CROSS_GATE")
            self.saveStage(1)

        if self.stage == "CROSS_GATE" and self.crossedGate(obs):
            self.greenMask = self.gate.lineMask
            self.stage = "TO_GOAL"
            self.get_logger().info("GATE crossed -> TO_GOAL (green is now a wall)")
            self.saveStage(2)

        # cross the gate if has it
        if self.stage == "CROSS_GATE":
            ax, ay, px, py, thru = self.gatePoints()
            nearApproach = math.hypot(obs.x - ax, obs.y - ay) < APPROACH_TOL
            aligned = abs(math.atan2(math.sin(obs.yaw - thru), math.cos(obs.yaw - thru))) < ALIGN_TOL
            if self.crossStep == "APPROACH" and nearApproach and not aligned:
                self.get_logger().info("GATE: at approach point, turning to face the gate")
                return (obs.x, obs.y, thru)
            if self.crossStep == "APPROACH" and nearApproach and aligned:
                self.crossStep = "PASS"
            if self.crossStep == "PASS":
                self.get_logger().info(f"GATE PASS: -> ({px:.2f},{py:.2f})")
                return (px, py, thru)

        # get walls and bounds for plan
        extra = np.zeros(g.lables.shape, dtype=bool)
        if self.redMask is not None:
            extra |= self.redMask
        if self.stage == "TO_GOAL":
            extra |= self.greenMask
        elif self.stage == "CROSS_GATE":
            extra |= self.gate.lineMask
        elif green is not None:
            extra |= green.lineMask
        p.biuldBlocked(extra)

        # plan with known walls and gates pulled from above
        start = g.world2grid(obs.x, obs.y)
        if not p.flooded(start):
            esc = p.escapeCell(start)
            if esc is None:
                self.get_logger().warn("RECOVERY: no safe cell near -> turning")
                return self.turn2Open(obs)
            ex, ey = g.grid2world(*esc)
            if math.hypot(ex - obs.x, ey - obs.y) < ESCAPE_MIN:
                p.flooded(esc)
            else:
                self.get_logger().info(f"RECOVERY: escaping to safe cell ({ex:.2f}, {ey:.2f})")
                return (ex, ey, math.atan2(ey - obs.y, ex - obs.x))

        # particurl plans for speshal stages
        if self.stage == "CROSS_GATE":
            wp = self.approachGate(obs)
            if wp is not None:
                return wp # cant reach goal point yet explore a bit more to find it

        target, why = None, "FRONTIER"
        if self.stage == "TO_GOAL":
            target, why = self.goalTarget(), "GOAL"
        if target is None:
            target, why = self.pickFrontier(obs), "FRONTIER"
        if target is None:
            self.get_logger().info("PLAN: no target looking around")
            self.guideCell = None
            return self.turn2Open(obs)

        self.setGuide(target)
        wps = p.simplify(p.path2(target))
        return self.wpFromPlan(obs, wps, why)

    def setGuide(self,cell):
        if cell != self.guideCell:
            gx, gy = self.grid.grid2world(*cell)
            self.decisionLog.append((gx, gy, "G"))
        self.guideCell = cell

    def saveStage(self, idx):
        if not DEBUG_MAP:
            return
        self.grid.saveDebugImg(f"map_{idx}.ppm", goal=GOAL_XY)
        self.grid.saveDecisionImg(f"decisions_{idx}.ppm", self.decisionLog, goal=GOAL_XY)

    def gatePoints(self):
        G, s = self.gate, self.nearSign
        ax, ay = G.cx + s * APPROACH_D * G.nx, G.cy + s * APPROACH_D*G.ny
        px, py = G.cx - s * PASS_D * G.nx, G.cy - s * PASS_D * G.ny
        return ax,ay,px,py, math.atan2(py-ay,px-ax)

    def  crossedGate(self, obs):
        G = self.gate
        side = (obs.x - G.cx) * G.nx + (obs.y - G.cy) * G.ny
        return side * self.nearSign < -0.2

    def approachGate(self, obs):
        g, p = self.grid, self.planner
        ax, ay, px, py, thru = self.gatePoints()
        acell = g.world2grid(ax, ay)
        if not g.inBound(*acell) or p.dist[acell] < 0:
            self.get_logger().info("GATE: aproch point not reachable yet, exploring more ot find")
            return None
        self.setGuide(acell)
        wp = self.wpFromPlan(obs, p.simplify(p.path2(acell)), "GATE")
        if wp is not None and math.hypot(wp[0] - ax, wp[1] - ay) < 0.05:
            wp = (wp[0], wp[1], thru)
        return wp
    
    def goalTarget(self):
        g,p = self.grid, self.planner
        gx, gy = GOAL_XY
        # flood  out
        ra, ca = g.world2grid(gx - 0.5, gy - 0.5)
        rb, cb = g.world2grid(gx + 0.5, gy + 0.5)
        sub = p.dist[ra:rb + 1, ca:cb + 1]
        if not (sub >= 0).any():
            return None         # no know path yet
        # cant reach so find nearest reachable
        big = np.where(sub >= 0, sub, np.iinfo(np.int32).max)
        r,c = np.unravel_index(np.argmin(big), sub.shape)
        return (ra + int(r), ca + int(c))

    def isBlacklist(self, cell):
        lim = RL_RADIUS / self.grid.res # convert m to cells
        return any(math.hypot(cell[0] - b[0], cell[1] - b[1]) <= lim for b in self.blacklist)

    def pickFrontier(self, obs):
        g, p = self.grid, self.planner
        gx, gy = GOAL_XY
        keepCells = KEEP_R / g.res
        best, bestScore = None, -math.inf
        cur, curScore = None, -math.inf
        greenRC = np.argwhere(g.lables == g.GREEN_EDGE) if self.stage == "FIND_GATE" else None
        for cells in p.frontiers():
            hit = p.nearestReachable(cells)
            if hit is None:
                continue
            cell, d = hit
            travel = d * g.res
            if self.isBlacklist(cell):
                continue
            if travel < MIN_STEP:
                self.blacklist.append(cell)
                continue
            cx, cy = g.grid2world(*cell)
            score = (W_SIZE * len(cells)
                     - W_DIST * travel
                     - W_GOAL * math.hypot(gx - cx, gy - cy))
            if greenRC is not None and len(greenRC):
                dGreen = np.min(np.hypot(greenRC[:, 0] - cell[0], greenRC[:, 1] - cell[1])) * g.res
                score += W_GREEN / (1.0 + dGreen)
            if score > bestScore:
                best, bestScore = cell, score
            if (self.guideCell is not None and
                    math.hypot(cell[0] - self.guideCell[0], cell[1] - self.guideCell[1]) <= keepCells
                    and score > curScore):
                cur, curScore = cell, score
        if cur is not None and curScore + HYST >= bestScore:
            best, bestScore = cur, curScore
        if best is not None:
            bx, by = g.grid2world(*best)
            self.get_logger().info(f"FRONTIER pick ({bx:.2f},{by:.2f}) score={bestScore:.2f}")
        return best

    def wpFromPlan(self,obs,wps,why):
        g = self.grid
        pts = [g.grid2world(r,c) for r,c in wps]
        # drop points already in
        while pts and math.hypot(pts[0][0] - obs.x, pts[0][1] - obs.y) < MIN_STEP:
            pts.pop(0)
        if not pts:
            return self.turn2Open(obs) # arived look around
        tx, ty = pts[0]
        dx,dy = tx - obs.x, ty - obs.y
        L = math.hypot(dx,dy)
        legHeading = math.atan2(dy,dx)
        if L > MAX_LEG:
            # to far to go in one leg brake it up to be safe
            tx, ty = obs.x + dx / L * MAX_LEG, obs.y + dy / L * MAX_LEG
            theta = legHeading
        elif len(pts) > 1:
            # ariving at end of leg face next leg directoin
            theta = math.atan2(pts[1][1] - ty, pts[1][0] - tx)
        else:
            theta = legHeading # final point keep heading
        self.get_logger().info(
            f"PLAN {why}: -> ({tx:.2f}), ({ty:.2f}) theta = {math.degrees(theta):.0f}deg"
            f"corners_left = {len(pts)}"
        )
        return (tx, ty, theta)
            
            



    def tick(self):
        if self.phase != Phase.OBSERVE or not self.observations_ready():
            return
        if not self.action.server_is_ready():
            return
        perception = self.perceive(self.odom, self.scan, self.image,
                                   self.camera_info)
        if perception is None:
            return
        updated = self.update_state(self.belief, perception)
        if updated is None:
            return
        self.belief = updated
        if self.goal_reached(self.belief):
            self.phase = Phase.DONE
            self.get_logger().info('Goal reached according to observed state')
            return
        waypoint = self.plan_next_waypoint(
            self.belief, self.last_waypoint, self.last_result_code)
        if waypoint is None:
            return
        self.send_waypoint(*waypoint)

    def send_waypoint(self, x, y, theta):
        self.decisionLog.append((float(x), float(y), "W")) 
        goal = DriveToPose.Goal()
        goal.x, goal.y, goal.theta = float(x), float(y), float(theta)
        self.last_waypoint = (goal.x, goal.y, goal.theta)
        self.phase = Phase.WAIT_FOR_MOTION
        self.action.send_goal_async(goal).add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        try:
            handle = future.result()
            if not handle.accepted:
                self.get_logger().error('Waypoint rejected')
                self.phase = Phase.FAILED
                return
            handle.get_result_async().add_done_callback(self.on_result)
        except Exception as exc:
            self.get_logger().error(f'Could not send waypoint: {exc}')
            self.phase = Phase.FAILED

    def on_result(self, future):
        try:
            response = future.result()
            result = response.result
            code = result.code
            self.last_result_code = code
            self.get_logger().info(
                f"Waypoint: status = {response.status}, code={code}"
            )
            ok = (response.status == GoalStatus.STATUS_SUCCEEDED and result.success and code == "OK")

            if ok:
                self.failCount = 0
            elif code in ("BLOCKED", "STALLED"):
                self.failCount +=1
                if self.failCount >= MAX_FAILS:
                    self.get_logger().error(
                        f"{self.failCount} failures in a row - Stopping"
                    )
                    self.phase = Phase.FAILED
                    return    
            else:
                self.get_logger().error(f"UNexpected result: code {code!r} - Stopping")
                self.phase = Phase.FAILED
                return

            self.observe_after = time.monotonic()
            self.phase = Phase.OBSERVE
        except Exception as exc:
            self.get_logger().error(f"Waypoint result: {exc}")
            self.phase = Phase.FAILED




def main():
    rclpy.init()
    node = PolicyExample()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
