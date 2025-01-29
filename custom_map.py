import pickle
import os
import neat
import pygame
import sys
from time import time
from car import Car

WIDTH = 1920
HEIGHT = 1060

CAR_SIZE_X = 20
CAR_SIZE_Y = 20

BORDER_COLOR = (255, 255, 255, 255)  # White
TRACK_COLOR = (0, 0, 0, 255)  # Black

MAX_DISTANCE = 300

START_POS = [830, 870]

# File to store the best model
MODEL_FILE = "genomes/best_genome.pkl"

config_path = "./config.txt"

current_generation = 0


def draw_map():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Draw the Map")

    clock = pygame.time.Clock()
    drawing = False
    setting_start = False
    setting_checkpoints = False
    map_surface = pygame.Surface((WIDTH, HEIGHT))
    map_surface.fill((255, 255, 255))
    checkpoints = []
    start_pos = None

    print("Draw the map with your mouse. Press 'S' to save and start the simulation.")
    print("Press 'P' to set the starting position, 'C' to add checkpoints.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if setting_start:
                    start_pos = pygame.mouse.get_pos()
                    print(f"Starting position set at: {start_pos}")
                    setting_start = False
                elif setting_checkpoints:
                    checkpoint_pos = pygame.mouse.get_pos()
                    pygame.draw.circle(map_surface, (255, 0, 0), checkpoint_pos, 40)
                    checkpoints.append(checkpoint_pos)
                else:
                    drawing = True

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    if start_pos is None:
                        print("Please set a starting position before saving.")
                    else:
                        pygame.image.save(map_surface, "custom_map.png")
                        print("Map saved as 'custom_map.png'. Starting simulation...")
                        running = False
                if event.key == pygame.K_p:
                    setting_start = True
                    print("Click on the map to set the starting position.")
                if event.key == pygame.K_c:
                    setting_checkpoints = True
                    print("Click to add checkpoints.")

        mouse_pos = pygame.mouse.get_pos()
        if drawing:
            pygame.draw.circle(map_surface, TRACK_COLOR, mouse_pos, 40)

        screen.blit(map_surface, (0, 0))
        if start_pos:
            pygame.draw.circle(screen, (0, 255, 0), start_pos, 10)  # Draw starting position
        for checkpoint in checkpoints:
            pygame.draw.circle(screen, (255, 0, 0), checkpoint, 40)  # Draw checkpoints

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return start_pos, checkpoints


def run_simulation(genomes, config):
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Cars")

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car(WIDTH, HEIGHT, MAX_DISTANCE, start_pos, CAR_SIZE_X, CAR_SIZE_Y, BORDER_COLOR))

    clock = pygame.time.Clock()
    raw_map = pygame.image.load("custom_map.png")
    game_map = pygame.transform.scale(raw_map, (WIDTH, HEIGHT)).convert()

    global current_generation
    current_generation += 1

    start = time()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

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

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()
            else:
                genomes[i][1].fitness *= 0.5

        if still_alive == 0 or time() - start > 10:
            running = False

        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    start_pos, checkpoints = draw_map()

    # Load Config
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            best_genome = pickle.load(f)
        print("Loaded best genome from file.")
        population = neat.Population(config)
        population.population[0] = best_genome
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(run_simulation, 1000)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved.")
