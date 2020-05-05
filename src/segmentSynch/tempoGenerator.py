import random
import sys


if len(sys.argv) > 1:
    if  '-h' in sys.argv[1] or '-help' in sys.argv[1]:
        print('If input given it has to be the start tempo!')
        sys.exit()

if len(sys.argv) > 1:
    currTempo = int(sys.argv[1])
else:
    currTempo=  int(input('Starting tempo?   '))

while True:
    _ = input('Press for the next tempo')

    currTempo += random.randint(-3, 3)
    print(currTempo)
