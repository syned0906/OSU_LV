numbers=[]
average=0
sum=0

while True:
    number=input("Unesite broj:")
    if number =="Done":
        break
        
    else:
        try:
            numbers.append(float(number))
        except:
         print('Uneseno je slovo a ne broj,unesite broj')
count=len(numbers)
for number in numbers:
    sum=sum+number
    average=sum/count

print('Najveci broj je:',max(numbers))
print('Najmanji broj je:',min(numbers))
print('Prosjek je:',average)
print('Uneseni brojevi:',count)
numbers.sort()
print(numbers)

    


    