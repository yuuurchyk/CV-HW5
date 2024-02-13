import os


from cvhw5.utils.env import get_exps_root


def get_exp_folder(exp_name: str) -> str:
    exp_path = os.path.join(get_exps_root(), exp_name)

    assert not os.path.exists(exp_path), f'experiment folder {exp_path} already exists!'

    os.makedirs(exp_name)

    return exp_path
