import sys, random
data = []
with open(sys.argv[1], "r") as f:
    for line in f:
        data.append(line)

random.shuffle(data)
n = int(len(data)*0.8)
train = data[:n]
test = data[n:]
print(len(train), len(test))

with open("vesa-train.txt", "w") as o:
    for line in train:
        o.write(line)

with open("vesa-test.txt", "w") as o:
    for line in test:
        o.write(line)
