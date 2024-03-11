import random

TABLE_SIZE = 5

START_POS = 0;
END_POS = TABLE_SIZE-1

cities = []
pointsDeDepart = {}

def initCities():
    global cities
    global pointsDeDepart
    cities = []
    processed = []
    pointsDeDepart = {}
    
    for i in range(0, TABLE_SIZE):
        cities.insert(i, [])
        for j in range(0, TABLE_SIZE):
            if (random.randint(0, 10) == 0) or (i == START_POS and j == START_POS) or (i == END_POS and j == END_POS):
            # if (i == 4 and j == 3) or (i == 6 and j == 6) or (i == 9 and j == 3)or (i == 6 and j == 9) or (i == 9 and j == 90 )or (i == 0 and j == 9) or (i == 0 and j == 0):
                cities[i].insert(j, 'ville')
            else:
                cities[i].insert(j, 'rien')
    
    
    for i in range(0, TABLE_SIZE):
        for j in range(0, TABLE_SIZE):
            if cities[i][j] == 'ville':
                for x in range(max([0, i-5]), min([i+5, TABLE_SIZE])):
                    for y in range(max([0, j-5]), min([j+5, TABLE_SIZE])):
                        string = (i*j)+(x*y)
                        if (i != x or j != y) :
                            if cities[x][y] == 'ville':
                                processed.append(string)
                                distance = (max([i, x]) - min([i, x]) + max([j, y]) - min([j, y]))
                                
                                if f'{i}{j}' not in pointsDeDepart.keys(): 
                                    pointsDeDepart[f'{i}{j}'] = {}
                                
                                data = {
                                            'prev': {
                                                'x': i,
                                                'y': j
                                            },
                                            'next': {
                                                'x': x,
                                                'y': y
                                            },
                                            'distance': distance
                                        }
                                
                                # if f'{x}{y}' in pointsDeDepart[f'{i}{j}'].keys():
                                #     print('Continue')
                                #     continue
                                    
                                for xx in range(min([i, x]), max([i, x])):
                                    if cities[xx][j] != 'ville':
                                        cities[xx][j] = distance
                                        pointsDeDepart[f'{i}{j}'][f'{x}{y}'] = data
                                
                                for yy in range(min([j, y]), max([j+1, y+1])):
                                    if cities[x][yy] != 'ville':
                                        cities[x][yy] = distance
                                        pointsDeDepart[f'{i}{j}'][f'{x}{y}'] = data
                                        
    # for chemin in pointsDeDepart:
    #     print(chemin, pointsDeDepart[chemin])
    #     print('\n')
                                  
def findPaths():
    global cities, pointsDeDepart
    traversed = []
    
    print(pointsDeDepart)
    
    # pointsDeDepart qui partent du point de départ (s)
    chemins = pointsDeDepart[str(START_POS) + str(START_POS)]    
    traversed.append(str(START_POS) + str(START_POS))
    print('00')
    indexChemin = '00'
    
    # Pour tous les chemins qui partent du départ
    while (indexChemin != str(END_POS) + str(END_POS)) :
        path = '00'
        
        for index in chemins:
            chemin = chemins[index]
            indexChemin = str(chemin['next']['x'])+str(chemin['next']['y'])
            
            if indexChemin == str(END_POS) + str(END_POS):
                print('Found !', path + ' => ' + str(indexChemin))
                path = '00'
                break
            
            if indexChemin not in traversed:
                path += ' => ' + indexChemin
                traversed.append(indexChemin)
                print(indexChemin, indexChemin == '44')
                chemins = pointsDeDepart[indexChemin]
                continue
    
def display():
    global cities
    
    buffer = '   '
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
    print('_ vide\n')

        
initCities()
display()
findPaths()