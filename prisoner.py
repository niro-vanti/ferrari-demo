import streamlit as st
import numpy as np
import pandas as pd

st.header('play me')
st.subheader('if you dare')
st.text('------------------------------------------------------------------------------------------')
st.image('assets/Images/table.png')
with st.expander('the rules'):
    st.write('Here are basic rules of the game. You must choose between playing a circle or a square. Your objective as '
             'the player is to maximize your payoff. You have a counterpart who faces the same choice and has the same '
             'objective. Your outcome—what you end up winning for yourself—depends on how your choice matches up with '
             'your counterpart’s.')
    st.write('The preceding table shows the payouts for each possible outcome. For example, if you were to play Circle '
             'and your counterpart plays Square, you will receive \$0 and your counterpart will receive \$10. Or if you '
             'both play Square you will both receive \$2.')
    st.write('Remember, your sole objective is maximizing your own payoff. It is neither competitive nor cooperative. '
             'You’re not motivated to help or hurt your counterpart. If they do well, that’s fine. If they don’t do well, '
             'that’s fine, as well. And just like you, your counterpart is solely focused on their own outcome.')
with st.expander('strategies'):
    st.write('***Always circle***: Always chooses Circle')
    st.write('***Always square***: Always chooses Square')
    st.write('***Grim trigger***: Start with Circle until they see a Square from you; then always chooses Square '
             'after that.')
    st.write('***Grudge***: Start with Circle until they see a Square from you; then plays Square at least twice '
             'until they see two successive Circles from you.')
    st.write('***Random***: Equal chance of choosing Circle or Square on any round, regardless of whatever you have '
             'chosen before.')
    st.write('***Slow to anger***: Start with Circle and continues doing so unless they see two (or more) successive '
             'Squares from you, after which they choose Square. They will return to Circle, right after seeing that '
             'you have done so in the most recent round.')
    st.write('***Slow to trust***: Start with two Squares and continues doing so until they see that you have chosen '
             'Circle in the two most recent rounds. In that case, they choose Circle. They will revert to their '
             'initial strategy whenever you play a Square.')
    st.write('***Tit-for-tat***: They start with Circle, and thereafter always mirrors your most recent choice.')

# constants

cols = ['your move', 'your reward', 'computer move', 'computer reward']
log_cols = ['your move', 'computer move']
reward_cols = ['your reward', 'computer reward']
values = [
    [[6, 6], [0, 10]],
    [[10, 0], [2, 2]]
]
possible_moves = ['circle', 'square']
strategies = ['Always circle', 'Always square', 'Grim trigger', 'Grudge', 'Random', 'Slow to anger', 'Slow to trust',
              'Tit-for-tat']


# funcs
def input_move(move):
    if st.session_state.game_round <= st.session_state.n_rounds:
        if move == 'circle':
            user_move = 0
            comp_move = np.random.choice([0, 1])
        elif move == 'square':
            user_move = 1
            comp_move = np.random.choice([0, 1])
        else:
            st.error('sdfsdf')
            user_move = 0
            comp_move = 0

        new = pd.DataFrame(columns=cols, index=[st.session_state.game_round])
        new['your move'] = possible_moves[user_move]
        new['your reward'] = values[user_move][comp_move][0]
        new['computer move'] = possible_moves[comp_move]
        new['computer reward'] = values[user_move][comp_move][1]
        st.session_state.state_table = pd.concat([st.session_state.state_table, new])
        st.session_state.game_round = st.session_state.game_round + 1
    else:
        st.session_state.game_over = True


def reset():
    st.session_state.game_round = 0
    st.session_state.game_over = False
    st.session_state.n_rounds = np.random.choice([10, 11, 12, 13, 14, 15])
    table = pd.DataFrame(columns=cols)
    table.index.columns = ['round']
    st.session_state.state_table = table
    st.session_state.strategy = strategy = np.random.choice(strategies)


# init
if 'n_rounds' not in st.session_state:
    st.session_state.n_rounds = np.random.choice([10, 11, 12, 13, 14, 15])
if 'state_table' not in st.session_state:
    table = pd.DataFrame(columns=cols)
    table.index.columns = ['round']
    st.session_state.state_table = table
if 'strategy' not in st.session_state:
    st.session_state.strategy = strategy = np.random.choice(strategies)
if 'game_round' not in st.session_state:
    st.session_state.game_round = 0
if 'game_over' not in st.session_state:
    st.session_state.game_over = False

# game
c1, c2 = st.columns(2)
circle_move = c1.button('circle me baby!', on_click=input_move, args=['circle'], key='circle')
square_move = c2.button('square me sucka!', on_click=input_move, args=['square'], key='square')
c1, c2 = st.columns(2)

log = c1.empty()
reward = c2.empty()

if st.session_state.game_over:
    st.error('game over buddy')
    reset = st.button('wanna go again?', on_click=reset, key='reset')
# circle_move = c1.button('circle me baby!', on_click=input_move, args=['circle'], key='circle')
# square_move = c2.button('square me sucka!', on_click=input_move, args=['square'], key='square')

with log.container():
    # st.write(st.session_state.strategy)
    st.table(st.session_state.state_table[log_cols])
with reward.container():
    st.table(st.session_state.state_table[reward_cols])
