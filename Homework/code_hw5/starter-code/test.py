f = open('cipher_text.txt','r').readlines()[0]
print(f)

shift = 6

fnew = f[shift:]
sum = 0
for i in range(len(fnew)):
    sum += int(fnew[i] == f[i])

print(sum)
