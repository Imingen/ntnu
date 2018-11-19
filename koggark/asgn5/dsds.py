    for i in range(-10, 0):
        if i < -6:
            x = grade(i, -10, -6, clip)
            tell += x * i
            nevn += x
            
        if i >= -6 and i < -2:
            clip = 0.2
            x = grade(i, -7, -3, clip)
            tell += x * i
            nevn += x
        if i > -3 and i <=0:
            clip  = 0.5
            x = grade(i, -3, 0, clip)
            tell += x * i
            nevn += x

    for i in range(0, 10):
        if i > 0 and i < 3:
            clip = 0.5
            x = reverse_grade(i, 0, 3, clip)
            tell += x * i
            nevn += x
        if i >= 3 and i < 7:
            clip = 0.1
            x = grade(i, 3, 7, clip)
            tell += x * i
            nevn += x

    print('_____________', tell/nevn)
