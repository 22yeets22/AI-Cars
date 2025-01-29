import pickle
import os
import neat
import pygame
import sys
from time import time
from tkinter import Tk, Label
from car import Car

WIDTH = 1920
HEIGHT = 1060

CAR_SIZE_X = 20
CAR_SIZE_Y = 20

BORDER_COLOR = (255, 255, 255, 255)

MAX_DISTANCE = 300

START_POS = [830, 870]

# Init variables
MAP_NUMBER = 3

# File to store the best model
MODEL_FILE = "genomes/best_genome.pkl"

config_path = "./config.txt"

current_generation = 0


def run_simulation(genomes, config):
    nets = []
    cars = []

    window = Tk()
    window.title("AI Cars [Info]")
    window.geometry("400x150+400+300")  # Fixed geometry (width, height, x, y)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Cars")

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car(WIDTH, HEIGHT, MAX_DISTANCE, START_POS, CAR_SIZE_X, CAR_SIZE_Y, BORDER_COLOR))

    clock = pygame.time.Clock()
    raw_map = pygame.image.load(f"maps/map{MAP_NUMBER}.png")
    game_map = pygame.transform.scale(raw_map, (WIDTH, HEIGHT)).convert()

    global current_generation
    current_generation += 1

    start = time()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # For each car get the acton it takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if car.speed >= 14:
                    car.speed -= 2  # Slow down
            else:
                car.speed += 2  # Speed up

        # Check if car is still alive and increase fitness if yes and break loop if not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()
            else:
                genomes[i][1].fitness *= 0.5  # Death negative

        if still_alive == 0 or time() - start > 10:
            running = False

        # Render
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Tkinter
        for widget in window.winfo_children():
            widget.destroy()
        Label(window, text=f"Generation: {current_generation}", font=('Helvetica bold', 40)).pack()
        Label(window, text=f"Still Alive: {still_alive}", font=('Helvetica bold', 40)).pack()
        window.update()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    window.destroy()

    # Save best genome every 50 generations
    if current_generation % 50 == 0:
        best_genome = max(genomes, key=lambda g: g[1].fitness)[1]  # Get the best genome
        with open(f"checkpoint_gen{current_generation}.pkl", "wb") as f:
            pickle.dump(best_genome, f)
        print(f"Checkpoint saved at generation {current_generation}")


if __name__ == "__main__":
    # Load Config
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Try to load a saved genome
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            best_genome = pickle.load(f)
        print("Loaded best genome from file.")
        population = neat.Population(config)
        population.population[0] = best_genome  # Use saved best genome
    else:
        # Create new population
        population = neat.Population(config)

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run Simulation
    winner = population.run(run_simulation, 1000)

    # Save the best genome
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved.")
