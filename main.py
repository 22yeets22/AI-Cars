import math
import neat
import pygame
import sys
import os
from time import time
from tkinter import Tk, Label

from constants import *

current_generation = 0


class Car:
    def __init__(self, start_speed=START_SPEED, image=CAR_IMAGE, radar_limit=RADAR_LIMIT):
        self.radar_limit = radar_limit
        self.sprite = pygame.image.load(image).convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))

        self.position = [830, 870]
        self.angle = 0
        self.speed = start_speed
        self.center = self.calculate_center()
        self.corners = []

        self.radars = []
        self.alive = True
        self.on_checkpoint = False
        self.checkpoint_rewards = 0
        self.rewards = 0
        self.time = 0

    def calculate_center(self):
        return [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

    def update(self, game_map):
        """Update car state."""
        self.move()
        self.check_collision(game_map)
        self.check_checkpoint(game_map)
        self.update_radars(game_map)

    def move(self):
        """Move the car based on its current angle and speed."""
        radians = math.radians(360 - self.angle)
        self.position[0] += math.cos(radians) * self.speed
        self.position[1] += math.sin(radians) * self.speed
        self.center = self.calculate_center()
        self.corners = self.calculate_corners()
        for corner in self.corners:
            if not self.point_inbounds(corner):
                self.alive = False
                return

    def calculate_corners(self):
        """Calculate the four corners of the car for collision detection."""
        length = 0.5 * CAR_SIZE_X
        angles = [30, 150, 210, 330]
        return [
            [
                self.center[0] + math.cos(math.radians(360 - (self.angle + a))) * length,
                self.center[1] + math.sin(math.radians(360 - (self.angle + a))) * length
            ]
            for a in angles
        ]

    def check_collision(self, game_map):
        """Determine if the car collides with borders."""
        for point in self.corners:
            if self.point_inbounds(point) and game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                return

    def check_checkpoint(self, game_map):
        """Check if the car is on a checkpoint."""
        if not self.point_inbounds((int(self.center[0]), int(self.center[1]))):
            self.alive = False
            return
        
        point_color = game_map.get_at((int(self.center[0]), int(self.center[1])))
        if not self.on_checkpoint and point_color == CHECKPOINT_COLOR:
            self.checkpoint_rewards += CHECKPOINT_REWARD
            self.on_checkpoint = True
        else:
            self.on_checkpoint = False

    def update_radars(self, game_map):
        """Update radar distances."""
        self.radars = []
        radar_angles = [-90, -45, 0, 45, 90, 135, 225]
        for angle in radar_angles:
            self.radars.append(self.calculate_radar(angle, game_map))

    def calculate_radar(self, angle_offset, game_map):
        """Calculate radar distance for a specific angle."""
        angle_rad = math.radians(360 - (self.angle + angle_offset))
        for length in range(1, self.radar_limit + 1):
            x = self.center[0] + math.cos(angle_rad) * length
            y = self.center[1] + math.sin(angle_rad) * length
            if not self.point_inbounds((x, y)) or game_map.get_at((int(x), int(y))) == BORDER_COLOR:
                return [(x, y), length]
        return [(x, y), self.radar_limit]

    def draw(self, screen):
        """Draw the car and its radars."""
        screen.blit(pygame.transform.rotate(self.sprite, self.angle), self.position)
        for radar in self.radars:
            pygame.draw.line(screen, RADAR_COLOR, self.center, radar[0], 1)
            pygame.draw.circle(screen, RADAR_COLOR, (int(radar[0][0]), int(radar[0][1])), 5)

    def get_data(self):
        """Return data for the AI model."""
        # Return the distance of the seven radars
        model_data = [0, 0, 0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            model_data[i] = int(radar[1] / 30)
        
        # Let the AI know its speed and angle
        model_data.extend([int(self.speed), int(self.angle)])
        return model_data

    def get_reward(self):
        """Calculate the fitness reward."""
        return self.rewards + self.checkpoint_rewards

    def is_alive(self):
        """Check if the car is still alive."""
        return self.alive

    @staticmethod
    def point_inbounds(point):
        """Check if a point is within the game boundaries."""
        return 0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT
    

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
        cars.append(Car())
    
    clock = pygame.time.Clock()
    raw_map = pygame.image.load(f"maps/map{MAP_NUMBER}.png")
    game_map = pygame.transform.scale(raw_map, (WIDTH, HEIGHT)).convert()

    global current_generation
    current_generation += 1

    frames = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # For each car get the action it takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += TURN_AMOUNT  # Left
            elif choice == 1:
                car.angle -= TURN_AMOUNT  # Right
            elif choice == 2:
                if car.speed >= MIN_SPEED:
                    car.speed -= SPEED_AMOUNT  # Slow down
            elif choice == 3:
                if car.speed <= MAX_SPEED:
                    car.speed += SPEED_AMOUNT  # Speed up

        # Check if car is still alive and increase fitness if yes and break loop if not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()
            else:
                # While the car is dead
                genomes[i][1].fitness -= DEATH_PUNISHMENT  # Death negative

        if still_alive == 0 or frames >= GENERATION_TIME_LIMIT * FPS:
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
        clock.tick(FPS)
        frames += 1

    pygame.quit()
    window.destroy()


def save_model(population, generation):
    """Saves the NEAT population to a file."""
    file_path = os.path.join(SAVE_DIR, f"neat_model_gen_{generation}.pkl")
    with open(file_path, "wb") as f:
        neat.Checkpointer.save_checkpoint(population, None, f)
    print(f"Model saved for generation {generation} at {file_path}")


if __name__ == "__main__":
    # Make the directory to save the models
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)
    
    # Create population and reporters
    # population = neat.Population(config)
    population = neat.Checkpointer.restore_checkpoint("models/neat-checkpoint-79")

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Add a custom checkpoint saver to trigger every 20 generations
    checkpointer = neat.Checkpointer(generation_interval=10, filename_prefix=f"{SAVE_DIR}/neat-checkpoint-")
    population.add_reporter(checkpointer)

    # Run simulation
    population.run(run_simulation, MAX_GENERATIONS)
