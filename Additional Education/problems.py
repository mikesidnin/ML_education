a = str(input())

part = ''
rezz = ''

for i in range(len(a)):
    count = 0
    if a[i] == a[i-1] and i != 0:
        continue
    for j in range(i, len(a)):
        if a[i] == a[j]:
            count += 1
        part = str(a[i]) + str(count)
    rezz = rezz + part
print(rezz)





a = int(input())
rez = [[0 for j in range(a)] for i in range(a)]
turn = 2 * a
for turn in range(turn):
    i = 0
    j = 0
    if turn % 4 == 0:
        i = turn // 4
        for j in range(i, a - i):
            rez[i][j] = rez[i][j-1] + 1
    if turn % 4 == 1:
        j = a - 1 - turn // 4
        for i in range(a - j, j + 1):
            rez[i][j] = rez[i - 1][j] + 1
    if turn % 4 == 2:
        i = a - 1 - turn // 4
        for j in range(i, a - i - 1, -1):
            rez[i][j-1] = rez[i][j] + 1
    if turn % 4 == 3:
        j = turn // 4
        for i in range(a - j - 1, j + 1, - 1):
            rez[i - 1][j] = rez[i][j] + 1
for i in rez:
    for j in i:
        print(j, end=" ")
    print()

