import random
import csv

TABLE_SIZE = 50

START_POS = 3;
END_POS = TABLE_SIZE - 3

global cities
cities = []

def initCities():
    global cities
    cities = []
    processed = []
    
    for i in range(0, TABLE_SIZE):
        cities.insert(i, [])
        for j in range(0, TABLE_SIZE):
            if (random.randint(0, 10) == 0) or (i == START_POS and j == START_POS) or (i == END_POS and j == END_POS):
            # if (i == 4 and j == 3) or (i == 6 and j == 6) or (i == 9 and j == 3)or (i == 6 and j == 9) or (i == 9 and j == 90 )or (i == 0 and j == 9) or (i == 0 and j == 0):
                cities[i].insert(j, 'ville')
            else:
                cities[i].insert(j, 'rien')
    
    chemins = []
    
    for i in range(0, TABLE_SIZE):
        for j in range(0, TABLE_SIZE):
            if cities[i][j] == 'ville':
                for x in range(max([0, i-5]), min([i+5, TABLE_SIZE])):
                    for y in range(max([0, j-5]), min([j+5, TABLE_SIZE])):
                        string = (i*j)+(x*y)
                        if (i != x or j != y) and string not in processed :
                            if cities[x][y] == 'ville':
                                processed.append(string)
                                
                                for xx in range(min([i, x]), max([i, x])):
                                    if cities[xx][j] != 'ville':
                                        distance = (max([i, x]) - min([i, x]) + max([j, y]) - min([j, y])) - 1
                                        cities[xx][j] = distance

                                        chemins.append({
                                            'prev': {
                                                'x': i,
                                                'y': j
                                            },
                                            'next': {
                                                'x': x,
                                                'y': y
                                            },
                                            'distance': distance
                                        })
                                
                                for yy in range(min([j, y]), max([j+1, y+1])):
                                    if cities[x][yy] != 'ville':
                                        distance = (max([i, x]) - min([i, x]) + max([j, y]) - min([j, y])) - 1
                                        cities[x][yy] = distance

                                        chemins.append({
                                            'prev': {
                                                'x': i,
                                                'y': j
                                            },
                                              'next': {
                                                'x': x,
                                                'y': y
                                            },
                                            'distance': distance
                                        })
                            # else:
                            #     cities[x][y] = 'chemin'
                                  
def display():
    global cities
    
    buffer = '  '
    for i in range(0, TABLE_SIZE):
        buffer = buffer + str(i).ljust(3, ' ')
        
    print(buffer)
    
    for i in range(0, TABLE_SIZE):
        buffer = ''
        for j in range(0, TABLE_SIZE):
            if i == START_POS and j == START_POS:
                buffer = buffer + str('s').ljust(3, ' ')
            elif i == END_POS and j == END_POS:
                buffer = buffer + str('e').ljust(3, ' ')
            elif cities[i][j] == 'ville':
                buffer = buffer + str('◼').ljust(3, ' ')
            elif cities[i][j] == 'rien':
                buffer = buffer + str('_').ljust(3, ' ')
            else:
                buffer = buffer + str(cities[i][j]).ljust(3, ' ')
        
        print(str(i).ljust(3, ' ') + buffer)
    print('\n◼ ville')
    print('_ vide')

initCities()
display()