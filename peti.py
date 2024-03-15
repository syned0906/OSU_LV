ham_count = 0
ham_total= 0
spam_count = 0
spam_total = 0
exclamation_count = 0

file =open('SMSSpamCollection.txt')
for line in file:
    line = line.strip()
    if line.startswith('ham'):
        ham_count =ham_count+1 
        ham_total += len(line.split()[1:])
    elif line.startswith('spam'):
        spam_count =spam_count+1 
        spam_total += len(line.split()[1:])
        if line.endswith('!'):
            exclamation_count += 1
spam_avg_words = spam_total / spam_count
ham_avg_words = ham_total / ham_count


print('Prosjecan broj hamova',ham_avg_words)
print('Prosjecan broj spamova',spam_avg_words)
print('Broj spamova koje zavrsavaju sa usklicnikom', exclamation_count)
