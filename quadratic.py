import numpy as np

class QuadraticEqua:
	def __init__(self,a,b,c, LOW_BOUND = -1000, UP_BOUND = 1001, POPULATION_SIZE = 100, 
		CROSSOVER_P = 0.7, MUTATION_P = 0.3, DISTRIBUTION_INDEX = 15, TOURNAMENT_SIZE = 2):
		self.a = a
		self.b = b
		self.c = c
		self.LB = LOW_BOUND
		self.UB = UP_BOUND
		self.POP_SIZE = POPULATION_SIZE
		self.CROSSOVER_P = CROSSOVER_P
		self.MUTATION_P = MUTATION_P
		self.DISTRIBUTION_INDEX = DISTRIBUTION_INDEX
		self.TOURNAMENT_SIZE = TOURNAMENT_SIZE
		self.population = np.array([])
		self.scores = np.array([])

	def get_population(self):
		return self.population

	def get_scores(self):
		return self.scores

	def set_scores(self, scores):
		self.scores = scores

	def initialize_population(self):
		self.population = np.array([np.random.uniform(self.LB, self.UB) for i in range(self.POP_SIZE)])
		

	def calc_fitness_score(self, x):
		return np.abs(self.a*x*x + self.b*x + self.c)

	def evaluate(self, group):
		scores = np.array([])
		for x in group:
			score = self.calc_fitness_score(x)
			scores = np.append(scores, score)
		return scores

	def tournament_selection(self):
		parents = np.array([])
		NUM_ROUNDS = int(self.POP_SIZE / self.TOURNAMENT_SIZE)


		for i in range(self.TOURNAMENT_SIZE):
			temp_pop = self.population
			for j in range(NUM_ROUNDS):
				competitors_indices = np.random.choice(len(temp_pop),self.TOURNAMENT_SIZE,replace = False)
				competitors = np.array([temp_pop[competitors_indices[k]] for k in range(self.TOURNAMENT_SIZE)])
				temp_pop = np.delete(temp_pop, competitors_indices)
				competitors_scores = self.evaluate(competitors)
				winner_index = np.argmin(competitors_scores)
				winner = competitors[winner_index]
				parents = np.append(parents, winner)
		return parents

	def check_possibility(self, p):
		return np.random.rand() < p

	def cross_over(self,parents):
		offsprings = np.array([])
		for i in range(int(self.POP_SIZE / 2)):

			father, mother = np.random.choice(parents,2,replace = False)

			if not self.check_possibility(self.CROSSOVER_P):
				offsprings = np.append(offsprings,[father, mother])
			else:
				u = np.random.rand()
				if u <= 0.5:
					beta = np.power(2*u, 1/(self.DISTRIBUTION_INDEX+1))
				else:
					beta = np.power(1/(2*(1-u)), 1/(self.DISTRIBUTION_INDEX+1))
				offspring0 = 0.5 * ((1+beta)*father + (1-beta)*mother)
				offspring1 = 0.5 * ((1-beta)*father + (1+beta)*mother)

				offsprings = np.append(offsprings, [offspring0, offspring1])

		return offsprings

	def mutate(self, offsprings):
		for i in range(len(offsprings)):
			if not self.check_possibility(self.MUTATION_P):
				continue
			else:
				u = np.random.rand()
				if u < 0.5:
					delta = np.power(2*u, 1/(self.DISTRIBUTION_INDEX+1)) - 1
				else:
					delta = 1 - np.power(2*(1-u), 1/(self.DISTRIBUTION_INDEX+1))
				offsprings[i] = offsprings[i] + (self.UB-self.LB)*delta
				
		return offsprings
	
	def select_survivors(self, parents, offsprings):
		temp_population = np.append(parents, offsprings)
		temp_scores = self.evaluate(temp_population)
		temp_population = self.sorted_population(temp_population, temp_scores)
		self.population = temp_population[:self.POP_SIZE]


	def sorted_population(self, population, scores):
		sorted_score_indices = np.argsort(scores)
		sorted_population = np.array([population[i] for i in sorted_score_indices])
		return sorted_population

	def get_N_solutions(self, N=0):
		if N<=0: N = self.POP_SIZE
		pop = self.sorted_population(self.population, self.scores)
		return pop[:N]
		# print(np.sort(self.scores)[:n])


def inp():
	res = input('Enter a, b, c: ').split()
	res = [int(val) for val in res]
	return res[0], res[1], res[2]

def main():
	a,b,c = inp()
	TERMINATION = 50
	solver = QuadraticEqua(a, b, c)
	solver.initialize_population()

	for i in range(TERMINATION):
		population = solver.get_population()
		scores = solver.evaluate(population)
		solver.set_scores(scores)

		parents = solver.tournament_selection()
		offsprings = solver.cross_over(parents)
		offsprings = solver.mutate(offsprings)
		solver.select_survivors(parents,offsprings)


	print(f'Problem: {a}x^2 + {b}x + {c} = 0: ')
	print('Solution: ', solver.get_N_solutions(2))

if __name__ == '__main__':
	main()
