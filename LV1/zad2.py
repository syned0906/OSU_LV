try:
    grade=float(input('unesite ocjenu:'))
    if grade <0.0 or grade >1.0:
        print("Ocjena nije unutar intervala [0.0-1.0]")
    else:
        if grade >=0.9:
            print('Ocjena je A')
        elif grade >=0.8 and grade<0.9:
            print('Ocjena je B')
        elif grade >=0.7 and grade<0.8:
            print('Ocjena je C')
        elif grade >=0.6 and grade<0.7:
            print('Ocjena je D')
        elif grade >=0.2 and grade <0.6:
            print('Ocjena je F')

except:
    print('Ocjena nije pravilno unesena')