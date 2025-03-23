import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
from collections import deque
from abc import ABC, abstractmethod

@dataclass
class SimulationSettings:
    num_agents: int = 10
    num_resources: int = 20
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    communication_range: float = 100
    resource_regeneration_rate: float = 0.01
    window_width: int = 1200
    window_height: int = 800
    simulation_width: int = 800
    simulation_height: int = 600
    collection_range: float = 15

class Vector2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def distance_to(self, other: 'Vector2D') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

class QTable:
    def __init__(self, num_states: int, num_actions: int):
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))
    
    def get_action(self, state: int, exploration_rate: float) -> int:
        state = max(0, min(state, self.num_states - 1))
        if random.random() < exploration_rate:
            return random.randint(0, self.num_actions - 1)
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, 
               learning_rate: float, discount_factor: float = 0.95):
        state = max(0, min(state, self.num_states - 1))
        next_state = max(0, min(next_state, self.num_states - 1))
        action = max(0, min(action, self.num_actions - 1))
        
        current_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
        self.q_table[state, action] = new_q

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (torch.cat(state), 
                torch.tensor(action), 
                torch.tensor(reward), 
                torch.cat(next_state))
    
    def __len__(self):
        return len(self.buffer)

class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, step=1):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.step = step
        self.active = False
        self.handle_width = 10
        self.handle_rect = pygame.Rect(self.get_handle_x(), y, self.handle_width, height)
    
    def get_handle_x(self):
        val_range = self.max_val - self.min_val
        position_range = self.rect.width - self.handle_width
        return self.rect.x + (self.value - self.min_val) / val_range * position_range
    
    def update(self, mouse_pos, mouse_pressed):
        if mouse_pressed[0]:
            if self.handle_rect.collidepoint(mouse_pos):
                self.active = True
            if self.active:
                val_range = self.max_val - self.min_val
                position_range = self.rect.width - self.handle_width
                new_x = min(max(mouse_pos[0], self.rect.x), 
                          self.rect.x + self.rect.width - self.handle_width)
                self.value = self.min_val + (new_x - self.rect.x) / position_range * val_range
                if self.step != 0:
                    self.value = round(self.value / self.step) * self.step
                self.value = min(max(self.value, self.min_val), self.max_val)
                self.handle_rect.x = self.get_handle_x()
        else:
            self.active = False

    def draw(self, screen, font):
        pygame.draw.rect(screen, (100, 100, 100), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.handle_rect)
        label_text = font.render(f"{self.label}: {self.value:.2f}", True, (255, 255, 255))
        screen.blit(label_text, (self.rect.x, self.rect.y - 20))

class MetricsGraph:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.colors = {
            'average_reward': (255, 100, 100),
            'resource_conflicts': (100, 255, 100),
            'cooperative_actions': (100, 100, 255),
            'message_entropy': (255, 255, 100)
        }
    
    def draw(self, screen, metrics, font):
        pygame.draw.rect(screen, (50, 50, 50), self.rect)
        
        if not metrics:
            return
        
        for i in range(4):
            y = self.rect.y + i * self.rect.height // 3
            pygame.draw.line(screen, (100, 100, 100), 
                           (self.rect.x, y), 
                           (self.rect.x + self.rect.width, y))
        
        for metric_name, color in self.colors.items():
            points = []
            for i, m in enumerate(metrics):
                x = self.rect.x + (i / len(metrics)) * self.rect.width
                max_val = max(m[metric_name] for m in metrics)
                if max_val > 0:
                    y = self.rect.bottom - (m[metric_name] / max_val) * self.rect.height
                else:
                    y = self.rect.bottom
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, color, False, points, 2)
        
        y_offset = self.rect.y
        for metric_name, color in self.colors.items():
            text = font.render(metric_name, True, color)
            screen.blit(text, (self.rect.x, y_offset))
            y_offset += 20

class BaseAgent(ABC):
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.position = Vector2D(x, y)
        self.velocity = Vector2D((random.random() - 0.5) * 2, (random.random() - 0.5) * 2)
        self.energy = 100
        self.reward = 0
        self.known_resources = {}
        
    def get_state(self, nearest_resource: Vector2D = None) -> int:
        if nearest_resource is None:
            return 0
        
        dx = nearest_resource.x - self.position.x
        dy = nearest_resource.y - self.position.y
        angle = np.arctan2(dy, dx)
        state = int(((angle + np.pi) / (2 * np.pi) * 4) % 4)
        return max(0, min(state, 3))
    
    @abstractmethod
    def get_action(self, state, exploration_rate):
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, learning_rate):
        pass
    
    def move(self, width: int, height: int, resources: List['Resource'], 
             settings: SimulationSettings, frame_count: int):
        nearest_resource = None
        min_distance = float('inf')
        
        for resource in resources:
            dist = resource.position.distance_to(self.position)
            if dist < min_distance:
                min_distance = dist
                nearest_resource = resource.position
        
        state = self.get_state(nearest_resource)
        action = self.get_action(state, settings.exploration_rate)
        
        action_to_direction = [
            Vector2D(1, 0),   
            Vector2D(0, 1),   
            Vector2D(-1, 0),  
            Vector2D(0, -1)   
        ]
        
        target_velocity = action_to_direction[action]
        self.velocity.x = self.velocity.x * 0.95 + target_velocity.x * 0.05
        self.velocity.y = self.velocity.y * 0.95 + target_velocity.y * 0.05
        
        self.position.x = (self.position.x + self.velocity.x) % width
        self.position.y = (self.position.y + self.velocity.y) % height
        
        self.energy = max(0, self.energy - 0.1)
        
        return state, action

class QTableAgent(BaseAgent):
    def __init__(self, id: int, x: float, y: float):
        super().__init__(id, x, y)
        self.q_table = QTable(4, 4)
    
    def get_action(self, state, exploration_rate):
        return self.q_table.get_action(state, exploration_rate)
    
    def update(self, state, action, reward, next_state, learning_rate):
        self.q_table.update(state, action, reward, next_state, learning_rate)

class DQNAgent(BaseAgent):
    def __init__(self, id: int, x: float, y: float):
        super().__init__(id, x, y)
        self.state_size = 4
        self.action_size = 4
        self.hidden_size = 64
        self.batch_size = 32
        
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(10000)
        self.steps_done = 0
    
    def get_action(self, state, exploration_rate):
        if random.random() < exploration_rate:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self._encode_state(state)).unsqueeze(0)
            return self.policy_net(state_tensor).max(1)[1].item()
    
    def _encode_state(self, state):
        encoded = torch.zeros(self.state_size)
        encoded[state] = 1
        return encoded
    
    def update(self, state, action, reward, next_state, learning_rate):
        if len(self.memory) < self.batch_size:
            return
            
        state_tensor = torch.FloatTensor(self._encode_state(state)).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(self._encode_state(next_state)).unsqueeze(0)
        
        self.memory.push(state_tensor, action, reward, next_state_tensor)
        
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = transitions
        
        current_q_values = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1))
        next_q_values = self.target_net(batch_next_state).max(1)[0].detach()
        expected_q_values = batch_reward + (0.99 * next_q_values)
        
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done += 1

class Resource:
    def __init__(self, x: float, y: float, value: float):
        self.position = Vector2D(x, y)
        self.value = value
        self.collected = False

class MARLSimulation:
    def __init__(self, settings: SimulationSettings = SimulationSettings()):
        self.settings = settings
        self.agents = []
        self.resources = []
        self.metrics = deque(maxlen=50)
        self.frame_count = 0
        
        pygame.init()
        self.screen = pygame.display.set_mode(
            (settings.window_width, settings.window_height)
        )
        pygame.display.set_caption("Multi-Agent Reinforcement Learning Simulation")
        
        self.font = pygame.font.Font(None, 24)
        
        self.sliders = [
            Slider(850, 50, 300, 20, 1, 50, settings.num_agents, "Number of Agents", 1),
            Slider(850, 120, 300, 20, 1, 100, settings.num_resources, "Number of Resources", 1),
            Slider(850, 190, 300, 20, 0, 1, settings.learning_rate, "Learning Rate", 0.01),
            Slider(850, 260, 300, 20, 0, 1, settings.exploration_rate, "Exploration Rate", 0.01),
            Slider(850, 330, 300, 20, 1, 100, settings.communication_range, "Communication Range", 1),
            Slider(850, 400, 300, 20, 0, 0.1, settings.resource_regeneration_rate, "Resource Regen Rate", 0.001)
        ]
        
        self.metrics_graph = MetricsGraph(50, 620, 700, 150)
        self.initialize_simulation()

    def initialize_simulation(self):
        self.agents = []
        num_agents = int(self.settings.num_agents)
        
        for i in range(num_agents):
            agent_type = random.random()
            x = random.random() * self.settings.simulation_width
            y = random.random() * self.settings.simulation_height
            
            if agent_type < 0.4:  
                agent = QTableAgent(id=i, x=x, y=y)
            elif agent_type < 0.7: 
                agent = DQNAgent(id=i, x=x, y=y)
            else:  
                agent = QTableAgent(id=i, x=x, y=y)
            
            self.agents.append(agent)
        
        self.resources = [
            Resource(
                x=random.random() * self.settings.simulation_width,
                y=random.random() * self.settings.simulation_height,
                value=50 + random.random() * 50
            )
            for _ in range(int(self.settings.num_resources))
        ]
    
    def handle_agent_interactions(self):
        collected_resources = []
        total_reward = 0
        cooperative_actions = 0
        resource_conflicts = 0
        
        for agent in self.agents:
            for resource in self.resources:
                if not resource.collected:
                    distance = resource.position.distance_to(agent.position)
                    if distance < self.settings.collection_range:
                        agent.energy += resource.value
                        agent.reward += resource.value
                        total_reward += resource.value
                        resource.collected = True
                        collected_resources.append(resource)
        
        self.resources = [r for r in self.resources if not r.collected]
        
        while len(self.resources) < self.settings.num_resources:
            if random.random() < self.settings.resource_regeneration_rate:
                self.resources.append(
                    Resource(
                        x=random.random() * self.settings.simulation_width,
                        y=random.random() * self.settings.simulation_height,
                        value=50 + random.random() * 50
                    )
                )
        
        return total_reward, cooperative_actions, resource_conflicts
    
    def update(self):
        self.settings.num_agents = int(self.sliders[0].value)
        self.settings.num_resources = int(self.sliders[1].value)
        self.settings.learning_rate = self.sliders[2].value
        self.settings.exploration_rate = self.sliders[3].value
        self.settings.communication_range = self.sliders[4].value
        self.settings.resource_regeneration_rate = self.sliders[5].value
        
        for agent in self.agents:
            try:
                state, action = agent.move(
                    self.settings.simulation_width,
                    self.settings.simulation_height,
                    self.resources,
                    self.settings,
                    self.frame_count
                )
                next_state = agent.get_state(None)
                agent.update(state, action, agent.reward, next_state, self.settings.learning_rate)
                agent.reward = 0
            except Exception as e:
                print(f"Error updating agent: {e}")
        
        total_reward, cooperative_actions, resource_conflicts = self.handle_agent_interactions()
        
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            avg_reward = total_reward / len(self.agents) if self.agents else 0
            self.metrics.append({
                'average_reward': avg_reward,
                'resource_conflicts': resource_conflicts,
                'cooperative_actions': cooperative_actions,
                'message_entropy': random.random()
            })
    
    def render(self):
        self.screen.fill((26, 26, 26))
        
        pygame.draw.rect(self.screen, (50, 50, 50), 
                        (0, 0, self.settings.simulation_width, self.settings.simulation_height), 2)
        
        for resource in self.resources:
            hue = int(resource.value)
            color = pygame.Color(0)
            color.hsla = (hue, 70, 50, 80)
            pygame.draw.circle(
                self.screen,
                color,
                (int(resource.position.x), int(resource.position.y)),
                5
            )
        
        for agent in self.agents:
            if isinstance(agent, QTableAgent):
                color = pygame.Color(255, 100, 100)
            elif isinstance(agent, DQNAgent):
                color = pygame.Color(100, 255, 100)
            else:
                color = pygame.Color(255, 100, 100)
                
            pygame.draw.circle(
                self.screen,
                color,
                (int(agent.position.x), int(agent.position.y)),
                5
            )
            
            energy_width = 20
            energy_height = 3
            energy_x = int(agent.position.x - energy_width/2)
            energy_y = int(agent.position.y - 10)
            pygame.draw.rect(self.screen, (50, 50, 50),
                           (energy_x, energy_y, energy_width, energy_height))
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (energy_x, energy_y, energy_width * (agent.energy/100), energy_height))
        
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()
        for slider in self.sliders:
            slider.update(mouse_pos, mouse_pressed)
            slider.draw(self.screen, self.font)
        
        self.metrics_graph.draw(self.screen, list(self.metrics), self.font)
        
        controls_text = "Controls: SPACE - Pause/Resume, R - Reset, ESC - Quit"
        text_surface = self.font.render(controls_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, self.settings.simulation_height + 20))
        
        legend_y = self.settings.simulation_height + 50
        self.screen.blit(self.font.render("Agent Types:", True, (255, 255, 255)), (10, legend_y))
        self.screen.blit(self.font.render("Q-Table", True, (255, 100, 100)), (120, legend_y))
        self.screen.blit(self.font.render("DQN", True, (100, 255, 100)), (220, legend_y))
        
        pygame.display.flip()
    
    def run(self):
        running = True
        paused = False
        clock = pygame.time.Clock()
        
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_r:
                            self.initialize_simulation()
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                
                if not paused:
                    self.update()
                
                self.render()
                clock.tick(60)
        
        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            pygame.quit()

if __name__ == "__main__":
    settings = SimulationSettings()
    simulation = MARLSimulation(settings)
    simulation.run()