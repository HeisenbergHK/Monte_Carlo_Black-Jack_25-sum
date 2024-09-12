import sys
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Custom Blackjack Environment
from gymnasium.envs.toy_text.blackjack import BlackjackEnv
from gymnasium import spaces


class CustomBlackjackEnv(BlackjackEnv):
    def __init__(self, natural=False):
        super().__init__(natural)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # player sum (0-31)
            spaces.Discrete(11),  # dealer card (1-10, Ace is 1)
            spaces.Discrete(2)))  # usable ace (0 or 1)

    def draw_card(self, np_random):
        return int(np_random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]))

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to player's hand and return
            self.player.append(self.draw_card(self.np_random))
            if sum_hand(self.player) > 25:
                return self._get_obs(), -1, True, False, {}
            return self._get_obs(), 0, False, False, {}
        else:  # stick: play out the dealer's hand, and score
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            return self._get_obs(), reward, True, False, {}


def cmp(a, b):
    return float(a > b) - float(a < b)


def sum_hand(hand):
    """Return the sum of a hand. Assumes that aces can be worth 1 or 11."""
    if usable_ace(hand) and sum(hand) + 10 <= 25:
        return sum(hand) + 10
    return sum(hand)


def usable_ace(hand):
    """Return true if the hand has a usable ace (i.e., one that can be worth 11 without busting)."""
    return 1 in hand and sum(hand) + 10 <= 25


def score(hand):
    """Return the score of a hand. The score is the highest sum <= 25, or the lowest score > 25."""
    return 0 if sum_hand(hand) > 25 else sum_hand(hand)


# Monte Carlo Control
def generate_episode_from_Q(env, Q, epsilon, nA):
    episode = []
    state, _ = env.reset()
    state = tuple(state)  # Ensure initial state is a tuple

    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
            if state in Q else env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        next_state = tuple(next_state)  # Ensure next state is a tuple

        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def get_probs(Q_s, epsilon, nA):
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s


def update_Q(env, episode, Q, alpha, gamma):
    states, actions, rewards = zip(*episode)

    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_Q)
    return Q


def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = env.action_space.n

    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start

    for i_episode in range(1, num_episodes + 1):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        epsilon = max(epsilon * eps_decay, eps_min)

        episode = generate_episode_from_Q(env, Q, epsilon, nA)

        Q = update_Q(env, episode, Q, alpha, gamma)

    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q


# Visualization
def plot_policy(policy):
    player_sum = np.arange(12, 26)  # Player's hand value range
    dealer_show = np.arange(1, 11)  # Dealer's visible card (1 to 10)

    policy_matrix = np.zeros((len(player_sum), len(dealer_show)))

    for (player, dealer, ace), action in policy.items():
        if 12 <= player <= 25:
            row = player - 12
            col = dealer - 1
            policy_matrix[row, col] = action

    fig, ax = plt.subplots()
    cax = ax.matshow(policy_matrix, cmap='coolwarm')

    for i in range(len(player_sum)):
        for j in range(len(dealer_show)):
            ax.text(j, i, f'{int(policy_matrix[i, j])}', va='center', ha='center')

    ax.set_xticks(np.arange(len(dealer_show)))
    ax.set_xticklabels(dealer_show)
    ax.set_yticks(np.arange(len(player_sum)))
    ax.set_yticklabels(player_sum)
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_title('Optimal Policy (0 = Stick, 1 = Hit)')

    fig.colorbar(cax)
    return fig, ax


class BlackjackGUI:
    def __init__(self, policy, Q):
        self.policy = policy
        self.Q = Q

    def show_policy_plot(self):
        fig_policy, ax_policy = plot_policy(self.policy)
        ax_policy.set_title('Optimal Policy (0 = Stick, 1 = Hit)', fontsize=12)
        ax_policy.set_xlabel('Dealer Showing', fontsize=10)
        ax_policy.set_ylabel('Player Sum', fontsize=10)

        # Adjusting font size of the text inside the matrix
        for text in ax_policy.texts:
            text.set_fontsize(8)

        plt.show()

    def show_value_plot(self):
        fig_value, ax_value = self.create_value_plot(self.Q)
        ax_value.set_title('Expected Return by Player Sum and Dealer Showing', fontsize=12)
        ax_value.set_xlabel('Player Sum', fontsize=10)
        ax_value.set_ylabel('Expected Return', fontsize=10)

        # Adjusting font size of the axis labels and legends
        ax_value.tick_params(axis='both', which='major', labelsize=8)
        ax_value.legend(fontsize=8)

        plt.show()

    def create_value_plot(self, Q):
        player_sum = np.arange(12, 26)
        dealer_show = np.arange(1, 11)
        usable_ace = [0, 1]  # 0 = No usable ace, 1 = Usable ace

        value_matrix = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))

        for player in player_sum:
            for dealer in dealer_show:
                for ace in usable_ace:
                    state = (player, dealer, ace)
                    if state in Q:
                        value_matrix[player - 12, dealer - 1, ace] = np.max(Q[state])

        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figsize for a smaller plot

        for i, ace in enumerate(usable_ace):
            for j, dealer in enumerate(dealer_show):
                ax.plot(player_sum, value_matrix[:, j, i], label=f'Dealer Showing = {dealer}, Usable Ace = {ace}')

        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Expected Return')
        ax.set_title('Expected Return by Player Sum and Dealer Showing')
        ax.legend(title='Dealer Showing and Ace')
        ax.grid(True)

        return fig, ax


if __name__ == "__main__":
    env = CustomBlackjackEnv()
    policy, Q = mc_control(env, 500_000, 0.02)

    gui = BlackjackGUI(policy, Q)
    gui.show_policy_plot()  # Show policy plot in a separate window
    gui.show_value_plot()   # Show value plot in another separate window