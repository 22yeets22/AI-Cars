import pygame
import math


class Car:
    def __init__(self, width, height, max_distance, start_pos, size_x, size_y, border_color):
        self.corners = []
        self.max_distance = max_distance

        self.size_x = size_x
        self.size_y = size_y

        # Note this is screen sizes
        self.width = width
        self.height = height

        self.sprite = pygame.image.load('car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (size_x, size_y))
        self.rotated_sprite = self.sprite

        self.position = pygame.Vector2(start_pos)
        self.angle = 0
        self.speed = 0

        self.speed_set = False

        self.center = [self.position[0] + size_x / 2, self.position[1] + size_y / 2]  # Calculate Center

        self.radars = []
        self.drawing_radars = []

        self.alive = True

        self.distance = 0
        self.time = 0

        self.border_color = border_color

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            int_point = (int(point[0]), int(point[1]))
            if self.point_inbounds(int_point) and game_map.get_at(int_point) == self.border_color:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        x, y = self.center[0], self.center[1]

        # Get the angle in radians
        angle_rad = math.radians(360 - (self.angle + degree))

        for length in range(1, self.max_distance + 1):
            if game_map.get_at((int(x), int(y))) == self.border_color:
                break

            x = self.center[0] + math.cos(angle_rad) * length
            y = self.center[1] + math.sin(angle_rad) * length

        dist = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], self.width - 120)

        self.distance += self.speed
        self.time += 1

        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], self.width - 120)

        self.center = [int(self.position[0]) + self.size_x / 2, int(self.position[1]) + self.size_y / 2]

        # Calculate the corners of the car
        length = 0.5 * self.size_x
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(game_map)
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        # Criteria: speed, distance, no crash
        return self.distance * self.speed  # i added speed here

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def point_inbounds(self, point):
        return 0 < point[0] < self.width and 0 < point[1] < self.height
