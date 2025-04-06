import gymnasium as gym
import matplotlib.pyplot as plt

def main():
    # Create the environment with render mode set to 'rgb_array'
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    # Reset the environment to the initial state
    observation, info = env.reset(seed=42)

    done = False
    total_reward = -100

    plt.ion()  # Turn on interactive mode for plotting
    fig, ax = plt.subplots()

    while not done:
        # Capture frame-by-frame
        frame = env.render()

        # Display the frame using matplotlib
        ax.clear()
        ax.imshow(frame)
        ax.axis('on')
        plt.pause(0.01)  # Pause for a brief period to simulate animation

        # Take a random action
        action = env.action_space.sample()

        # Step the environment forward
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if done:
            print("Episode finished. Total reward:", total_reward)
            break

    # Close the environment
    env.close()
    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    main()
