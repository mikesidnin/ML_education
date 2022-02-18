a = int(input())

if a == 0:
    print(a, "программистов")

if len(str(a)) > 2:
    new_a = a % 100
    ost_new_a = new_a % 10
    if new_a % 10 == 1 and new_a != 11:
        print(a, "программист")
    elif (new_a % 10 == 2 and new_a != 12) or (new_a % 10 == 3 and new_a != 13) or (new_a % 10 == 4 and new_a != 14):
        print(a, "программистa")
    elif 9 >= ost_new_a >= 5 or ost_new_a == 0 or 10 <= new_a <= 20:
        print(a, "программистов")
else:
    ost_new_a = a % 10
    if a % 10 == 1 and a != 11:
        print(a, "программист")
    elif (a % 10 == 2 and a != 12) or (a % 10 == 3 and a != 13) or (a % 10 == 4 and a != 14):
        print(a, "программистa")
    elif 9 >= ost_new_a >= 5 or ost_new_a == 0 or 10 <= a <= 20:
        print(a, "программистов")


# task 2
b = int(input())

first_b = str(b // 1000)
last_b = str(b % 1000)

sum_first = int(first_b[0]) + int(first_b[1]) + int(first_b[2])
sum_last = int(last_b[0]) + int(last_b[1]) + int(last_b[2])

if sum_last == sum_first:
    print("Счастливый")
else:
    print("Счастливый")

