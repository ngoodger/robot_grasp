from baselines.ppo1 import pposgd_simple
from baselines.ppo1 import mlp_policy
import niryo_env
from mpi4py import MPI
import baselines.common.tf_util as U
from baselines.common import set_global_seeds


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = (lcl['t'] > 100 and
                 sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199)
    return is_solved


def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
    return mlp_policy.MlpPolicy(name=name,
                                ob_space=ob_space,
                                ac_space=ac_space,
                                hid_size=64,
                                num_hid_layers=1)


def main():
    seed = 0
    sess = U.single_threaded_session()
    sess.__enter__()
    env = niryo_env.NiryoRobotEnv()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    # model = ppo1.models.mlp([16, 12])
    print(env.observation_space)
    print(env.action_space)
    act = pposgd_simple.learn(
              env,
              policy_fn,
              max_timesteps=int(100000000000),
              timesteps_per_actorbatch=2048,
              clip_param=0.2, entcoeff=0.00,
              optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
              gamma=0.99, lam=0.95,
              schedule='linear'
              #callback=callback
          )
    print("Saving model to balance.pkl")
    act.save("balance.pkl")


if __name__ == '__main__':
    main()
