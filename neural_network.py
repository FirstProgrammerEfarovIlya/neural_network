from numpy import exp, dot, array, random
from random import randint
import os

random.seed(1)

#преобразования ввода пользователя
def int_tuple(s):
	for i in range(len(s)):
		s[i] = int(s[i])
	s = tuple(s)
	return s

#сама нейроеть
class neural:
	#инициализация весов
	def __init__(self):
		self.weights = random.random((4, 1))

	#нормализация и маштабирование данных
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	#обучение сети, алгоритм одной итерации
	def train(self, inputs, outputs):
		output = self.think(inputs)
		errors = outputs - output
		adjustment = dot(inputs.T, errors * output * (1 - output))
		self.weights += adjustment

	#ответ нейросети, обдумывание результата
	def think(self, inputs):
		print("{0[0][0]} {0[1][0]} {0[2][0]} {0[3][0]}".format(self.weights))
		return self.__sigmoid(dot(inputs, self.weights))

	#генерация выборок для обучения сети
	@staticmethod
	def create_data():
		inputs = [[randint(0, 1) for j in range(4)]
		for j in range(20)]
		outputs = []

		for i in range(10):
			inputs[randint(0, len(inputs) - 1)][3] = 1

		for e in inputs:
			if e[3] == 1:
				outputs += [1]
			else:
				outputs += [0]

		inputs = array(inputs)
		outputs = array([outputs]).T

		return inputs, outputs

	#обучение сети
	def traininig(self, num):
		for i in range(num):
			inputs, outputs = self.create_data()
			self.train(inputs, outputs)

#восстановение памяти из текстового файла
def read_memory(neuron):
	with open("memory.txt", "r") as file:
		w = []
		for line in file:
			w += [float(line)]
		w = array([w]).T
		neuron.weights = w

#запись имеющейся памяти в текстовый файл, создание памяти
def write_memory(neuron):
	with open("memory.txt", "w") as file:
		for e in neuron.weights:
			file.write(str(e[0]) + "\n")



DONE = True
network = neural()
while(DONE):
	#поиск и восстановление имеющейся памяти
	for e in os.listdir():
		if e == "memory.txt":
			read_memory(network)

	try:
		ans = int(input("1 - exit, 2 - continue, 3 - train neural \n: "))
		if ans == 1:
			DONE = False
			print("Programm complete")
		elif ans == 2:
			ans = input("Enter data, please: ")
			ans = int_tuple(ans.split(" "))
			ans = network.think(array(ans))[0]
			if ans > 0.5:
				print("Yes, this is true combination!!!")
			else:
				print("No, this is false combination!!!")
			print(f"I am sure of this on {ans * 100 :f} %")
		elif ans == 3:
			ans = int(input("Enter count training: "))
			network.traininig(ans)
			write_memory(network)
			print("Neural network's train is complete")
	except ValueError:
		print("Error!!!")