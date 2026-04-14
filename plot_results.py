log = open("training_log.txt", "r")
csv = open("results.csv", "w")
csv.write("Episode,Points\n")
line = log.readline()

while line.startswith("Episode:"):
  episode = line.split()[1]
  points = line.split()[5]
  csv.write(episode + "," + points + "\n")
  line = log.readline()

while not line.startswith("Episode:"):
  line = log.readline()
  
while line.startswith("Episode:"):
  episode = line.split()[1]
  points = line.split()[5]
  csv.write(episode + "," + points + "\n")
  line = log.readline()