import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the Falling Objects")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 250, 0)
BLUE = (0, 0, 255)

# Game variables
basket_width, basket_height = 100, 20
basket_x = WIDTH // 2 - basket_width // 2
basket_y = HEIGHT - 50
basket_speed = 7

object_width, object_height = 30, 30
object_x = random.randint(0, WIDTH - object_width)
object_y = -object_height
object_speed = 5

score = 0
speed_increase_threshold = 3  # Change speed every 10 points
font = pygame.font.Font(None, 36)

# Clock
clock = pygame.time.Clock()

# Game loop
running = True
while running:
    screen.fill(WHITE)

    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get key states
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and basket_x > 0:
        basket_x -= basket_speed
    if keys[pygame.K_RIGHT] and basket_x < WIDTH - basket_width:
        basket_x += basket_speed

    # Update object position
    object_y += object_speed

    # Check if object hits the ground
    if object_y > HEIGHT:
        object_y = -object_height
        object_x = random.randint(0, WIDTH - object_width)

    # Check if basket catches the object
    if (basket_x < object_x < basket_x + basket_width or 
        basket_x < object_x + object_width < basket_x + basket_width) and \
       (basket_y < object_y + object_height):
        score += 1
        object_y = -object_height
        object_x = random.randint(0, WIDTH - object_width)

        # Increase object speed every 10 points
        if score % speed_increase_threshold == 0:
            object_speed += 2
            print(f"Speed increased to {object_speed}")

    # Draw the basket
    pygame.draw.rect(screen, BLUE, (basket_x, basket_y, basket_width, basket_height))

    # Draw the falling object
    pygame.draw.rect(screen, RED, (object_x, object_y, object_width, object_height))

    # Draw the score
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(30)

pygame.quit()
