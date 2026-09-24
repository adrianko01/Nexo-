"""
NEXO - prototipo experimental de aprendizaje con "vacío".

Idea:
    percibir -> predecir -> actuar -> observar -> medir error
    -> actualizar memoria -> actualizar incertidumbre -> repetir

El "vacío" NO es una ausencia accidental:
    vacío = incertidumbre sobre aspectos relevantes del entorno.

El agente intenta reducir incertidumbre mediante exploración,
pero permanece dentro de límites de seguridad del entorno simulado.

Este es un prototipo educativo, no AGI.
"""

from dataclasses import dataclass, field
import random
import math


@dataclass
class Memory:
    experiences: list = field(default_factory=list)

    def remember(self, state, action, predicted, actual, error):
        self.experiences.append({
            "state": state,
            "action": action,
            "predicted": predicted,
            "actual": actual,
            "error": error,
        })

        # Mantener memoria acotada en este prototipo.
        if len(self.experiences) > 5000:
            self.experiences.pop(0)


@dataclass
class Agent:
    # Creencias simples sobre el efecto de cada acción.
    beliefs: dict = field(default_factory=lambda: {
        -1: 0.0,   # izquierda
         0: 0.0,   # quedarse
         1: 0.0,   # derecha
    })

    uncertainty: dict = field(default_factory=lambda: {
        -1: 1.0,
         0: 1.0,
         1: 1.0,
    })

    memory: Memory = field(default_factory=Memory)

    learning_rate: float = 0.20
    exploration: float = 0.30

    def predict(self, action):
        return self.beliefs[action]

    def choose_action(self):
        # El "vacío" impulsa exploración:
        # cuanto mayor es la incertidumbre, más interesante resulta probar.
        scores = {}

        for action in self.beliefs:
            knowledge_value = self.beliefs[action]
            curiosity_value = self.uncertainty[action]
            scores[action] = knowledge_value + self.exploration * curiosity_value

        # Exploración ocasional para evitar quedar atrapado en una hipótesis.
        if random.random() < self.exploration:
            return random.choice(list(self.beliefs.keys()))

        return max(scores, key=scores.get)

    def learn(self, action, predicted, actual, state):
        error = actual - predicted

        # Actualización de la creencia.
        self.beliefs[action] += self.learning_rate * error

        # La incertidumbre disminuye cuando una experiencia confirma
        # razonablemente la predicción y aumenta cuando aparece sorpresa.
        surprise = min(abs(error), 1.0)
        self.uncertainty[action] = (
            0.90 * self.uncertainty[action]
            + 0.10 * surprise
        )

        self.memory.remember(
            state=state,
            action=action,
            predicted=predicted,
            actual=actual,
            error=error,
        )

        return error


class World:
    """
    Mundo extremadamente sencillo:
    cada acción tiene una consecuencia real desconocida al principio.
    """

    def __init__(self):
        self.true_effect = {
            -1: -0.8,
             0:  0.1,
             1:  0.7,
        }

    def step(self, action):
        noise = random.uniform(-0.05, 0.05)
        return self.true_effect[action] + noise


def run(steps=1000):
    world = World()
    agent = Agent()

    total_error = 0.0

    for t in range(steps):
        state = t

        # 1. Percibir/representar el estado.
        action = agent.choose_action()

        # 2. Predicción.
        predicted = agent.predict(action)

        # 3. Acción en el mundo.
        actual = world.step(action)

        # 4. Error.
        error = agent.learn(action, predicted, actual, state)
        total_error += abs(error)

        if (t + 1) % 100 == 0:
            print(f"\nCiclo {t + 1}")
            print("Creencias:", {
                a: round(v, 3)
                for a, v in agent.beliefs.items()
            })
            print("Vacío/incertidumbre:", {
                a: round(v, 3)
                for a, v in agent.uncertainty.items()
            })
            print("Error medio:",
                  round(total_error / (t + 1), 4))

    return agent


if __name__ == "__main__":
    agent = run()

    print("\n--- ESTADO FINAL ---")
    print("Experiencias almacenadas:", len(agent.memory.experiences))
    print("Creencias:", agent.beliefs)
    print("Incertidumbre:", agent.uncertainty)
