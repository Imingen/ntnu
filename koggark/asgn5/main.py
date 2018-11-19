def fuzzy_and(x1, y1):
    return min(x1, y1)

def fuzzy_or(x1, y1):
    return max(x1, y1)

def fuzzy_not(p):
    return 1.0 - p

def triangle(position, x0, x1, x2, clip):
    value = 0.0
    if position >= x0 and position <= x1:
        value = ((position - x0)/(x1 - x0)) 
    elif position >= x1 and position <= x2:
        value = ((x2 - position)/(x1 - x0))
    
    if value > clip:
        value = clip
    
    return value

def grade(position, x0, x1, clip):
    value = 0.0
    if position >= x1:
        value = 1.0
    elif position <= x0:
        value = 0.0
    else:
         value = ((position - x0)/(x1 - x0))

    if value > clip:
        value = clip
    
    return value

def reverse_grade(position, x0, x1, clip):
    value = 0.0
    if position <= x0:
        value = 1.0
    elif position >= x1:
        value = 0.0
    else:
        value = ((x1 - position)/(x1 - x0))

    if value > clip:
        value = clip

    return value

def rule_4():
    not_growing = fuzzy_not(growing)
    not_g_fast = fuzzy_not(0.0)
    tmp = fuzzy_or(not_growing, not_g_fast)
    return fuzzy_and(0.0, tmp)

if __name__ == "__main__":
    
    # fuzzification 
    # distance
    perfect = triangle(3.7, 3.5, 5., 6.5, 1.0)
    print(f"Perfect: {perfect}")
    small = triangle(3.7, 1.5, 3, 4.5, 1.0)
    print(f"Small: {small}")
    very_small = reverse_grade(3.7, 1, 2.5, 1.0)

    # delta
    growing = triangle(1.2, 0.5, 2, 3.5, 1.0)
    print(f"Growing: {growing}")
    stable = triangle(1.2, -1.5, 0.0, 1.5, 1.0)
    print(f"Stable: {stable}")


    # rule evaluation
    action_none = fuzzy_and(small, growing)
    action_slowdown = fuzzy_and(small, stable)
    action_speedup = fuzzy_and(perfect, growing)
    action_floorit = rule_4()
    action_brakehard = very_small

    clip = 0.0
    tell = 0
    nevn = 0
    tmp = 0
    for i in range(-6, -2):
        x = grade(i, -7, -2, 0.2)
        tmp += i
        nevn += x
    tell += tmp * x
    tmp = 0

    for i in range(-2, 0):
        x = grade(i, -3, -0, 0.5)
        tmp += i
        nevn += x
    tell += tmp * x
    tmp = 0

    for i in range(0, 2):
        x = reverse_grade(i, 3, 0, 0.5)
        tmp += i
        nevn += x
    tell += tmp * x
    tmp = 0

    for i in range(3, 6):
        x = reverse_grade(i, 3, 7, 0.1)
        tmp += i
        nevn += x
    tell += tmp * x
    tmp = 0

    print('_____________', tell/nevn)






