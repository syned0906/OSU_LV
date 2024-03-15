file=open('song.txt')
count=0
words={}

for line in file :
    line =line.rstrip()
    line=line.split()
    for word in line:
       word=word.lower().strip('!.?,')
       if word.isalpha():
        if word not in words:
             words[word]=1
        else:
           words[word]+=1
        
for word,value in words.items():
    if value == 1:
        print(word)
        count=count+1
        
print(count)

    

