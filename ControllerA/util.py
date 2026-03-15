#Author: Kyle Nguyen
#Date: September 2025
#Description: Utility functions for fuzzy logic controller and other controllers

import math
# triangular membership function

def triag(x, a, b, c):# slope magic
    if b ==a: return 0.0
    if c ==b: return 0.0
    if x <= a or x >= c:# outside the triangle
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a)  #on the upslope, linear interpolation from a to b
    if b < x < c:
        return (c - x) / (c - b)  #on the downslope, linear interpolation from b to c
    
    
def trap(x, a, b, c, d):# trapezoidal membership function
    if x <= a or x >= d:# outside the trapezoid
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)  #on the upslope, linear interpolation from a to b
    if b <= x <= c:
        return 1.0  #top of the trapezoid
    if c < x < d:
        return (d - x) / (d - c)  #on the downslope, linear interpolation from c to d

# makes the numbers wrap around -180 to +180
def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0




#try to guess where to shoot
def intercept_point(ship_pos, ship_vel, target_pos, target_vel):
    
    dx, dy = target_pos[0] - ship_pos[0], target_pos[1] - ship_pos[1] #vector from ship to target
    dvx, dvy = target_vel[0] - ship_vel[0], target_vel[1] - ship_vel[1]#relative vel, how the target is moving compared to us

    bullet_speed = 800.0
    #Quadratic problem in time t
    #LHS: squared distance to target at time t
    #RHS: squared distance bullet travels in time t
    #Solve for t, then use t to find intercept point
    # a*t^2 + b*t + c = 0
    a = dvx**2 + dvy**2 - bullet_speed**2
    b = 2 * (dx*dvx + dy*dvy)
    c = dx**2 + dy**2

    delta = b*b - 4*a*c
    if delta < 0 or abs(a) < 1e-6:
        return target_pos
    t1 = (-b + math.sqrt(delta)) / (2*a)
    t2 = (-b - math.sqrt(delta)) / (2*a)
    t_candidates = [t for t in (t1, t2) if t > 0]#keeps only positive times

    if not t_candidates:
        return target_pos
    #pick the soonest intercept time
    t = min(t_candidates) 
    return (target_pos[0] + target_vel[0]*t,#Starting at its current position, move it forward along its velocity vector for t seconds
            target_pos[1] + target_vel[1]*t)




def side_score(perpendicular, distance):
    #the closer to 0 the better, the closer to 1 the worse
    #perpendicular is how far off center the target is, in units of ship radius
    #distance is how far away the target is, in units of ship radius
    if distance < 1.0:
        return 0.0#right on top of us, so shoot it
    return min(1.0, (perpendicular / distance)**2)#the further away it is, the more we care about being centered on it


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def find_nearest_asteroid(ship_state, game_state):
    asteroids = getattr(game_state, "asteroids", [])
    if not asteroids:
        return None

    ship_pos = getattr(ship_state, "position", (0, 0))
    nearest = min(asteroids, key=lambda a: distance(ship_pos, getattr(a, "position", (0, 0))))
    return nearest

def angle_between(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))