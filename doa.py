import random
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load


class DOA_SVM:
    def __init__(self, pop, dim, maksiter, X_train, X_test, y_train, y_test):
        self.pop = pop
        self.dim = dim
        self.maksiter = maksiter
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.population = self.initialize_population()
        self.fitness = self.fitness_function(self.population)
        self.best_pop, self.best_member, self.best_fitness = self.best_population()

    def initialize_population(self):
        population = np.zeros((self.pop, self.dim))
        for i in range(self.pop):
            for d in range(self.dim):
                if d == 0:
                    lb, ub = 0.1, 100
                    population[i, d] = lb + random.uniform(0, 1) * (ub - lb)
                elif d == 1:
                    lb, ub = 0.01, 0.1
                    population[i, d] = lb + random.uniform(0, 1) * (ub - lb)
        return population

    def fitness_function(self, population):
        fitness = np.zeros(self.pop)
        for i in range(self.pop):
            C = population[i, 0]
            gamma = population[i, 1]
            model = SVC(kernel='rbf', C=C, gamma=gamma)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            fitness[i] = accuracy_score(self.y_test, y_pred)
        return fitness

    def best_population(self):
        best_fitness = self.fitness[0]
        best_member = self.population[0]
        best_pop = None
        for i in range(self.pop):
            if self.fitness[i] > best_fitness:
                best_fitness = self.fitness[i]
                best_member = self.population[i]
                best_pop = i
        return best_pop, best_member, best_fitness

    def exploration_doa(self):
        old_population = self.population.copy()
        old_fitness = self.fitness.copy()
        new_population = np.zeros((self.pop, self.dim))
        for i in range(self.pop):
            for d in range(self.dim):
                new_population[i, d] = old_population[i, d] + random.uniform(0, 1) * (
                    self.best_member[d] - random.choice([1, 2]) * old_population[i, d]
                )
                if d == 0:
                    new_population[i, d] = np.clip(new_population[i, d], 0.1, 100)
                elif d == 1:
                    new_population[i, d] = np.clip(new_population[i, d], 0.01, 0.1)
        new_fitness = self.fitness_function(new_population)
        for i in range(self.pop):
            if new_fitness[i] > old_fitness[i]:
                self.fitness[i] = new_fitness[i]
                self.population[i] = new_population[i]
            else:
                self.fitness[i] = old_fitness[i]
                self.population[i] = old_population[i]

    def exploitation_doa(self, t):
        old_population = self.population.copy()
        old_fitness = self.fitness.copy()
        new_population = np.zeros((self.pop, self.dim))
        for i in range(self.pop):
            for d in range(self.dim):
                if d == 0:
                    lb = 0.1
                    ub = 100
                    new_population[i, d] = old_population[i, d] + (1 - random.uniform(0, 1)) * ((ub - lb) / t)
                    new_population[i, d] = np.clip(new_population[i, d], lb, ub)
                elif d == 1:
                    lb = 0.01
                    ub = 0.1
                    new_population[i, d] = old_population[i, d] + (1 - random.uniform(0, 1)) * ((ub - lb) / t)
                    new_population[i, d] = np.clip(new_population[i, d], lb, ub)
        new_fitness = self.fitness_function(new_population)
        for i in range(self.pop):
            if new_fitness[i] > old_fitness[i]:
                self.fitness[i] = new_fitness[i]
                self.population[i] = new_population[i]
            else:
                self.fitness[i] = old_fitness[i]
                self.population[i] = old_population[i]

    def run(self):
        print(f"Initialization: Best Fitness = {self.best_fitness}")
        print(f"Best C = {self.best_member[0]:.4f}, Best Gamma = {self.best_member[1]:.4f}")

        for t in range(1, self.maksiter + 1):
            previous_population = self.population.copy()
            previous_fitness = self.fitness.copy()
            self.exploration_doa()
            self.exploitation_doa(t)
            self.best_pop, self.best_member, self.best_fitness = self.best_population()
            print(f"\nIteration {t}: Best Fitness = {self.best_fitness}")
            print(f"Best C = {self.best_member[0]:.4f}, Best Gamma = {self.best_member[1]:.4f}")
            for i in range(self.pop):
                if self.fitness[i] > previous_fitness[i]:
                    self.population[i] = self.population[i]
                    self.fitness[i] = self.fitness[i]
                else:
                    self.population[i] = previous_population[i]
                    self.fitness[i] = previous_fitness[i]
            self.best_pop, self.best_member, self.best_fitness = self.best_population()
        model = SVC(kernel='rbf', C=self.best_member[0], gamma=self.best_member[1])
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)

        print("\nFinal Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap='Blues')
        dump(model, 'best_doa_svm_model.pkl')
        print("Best Model Saved as 'best_doa_svm_model.pkl'")
        return self.best_fitness, self.best_member