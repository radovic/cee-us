# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .franka_two_cubes_move import FrankaTwoCubesMove
from .franka_cube_move import FrankaCubeMove

def resolve_allegro_kuka(cfg, *args, **kwargs):
    subtask_name: str = cfg["env"]["subtask"]
    subtask_map = dict(
        reorientation=AllegroKukaReorientation,
        throw=AllegroKukaThrow,
        regrasping=AllegroKukaRegrasping,
    )

    if subtask_name not in subtask_map:
        print("!!!!!")
        raise ValueError(f"Unknown subtask={subtask_name} in {subtask_map}")

    return subtask_map[subtask_name](cfg, *args, **kwargs)

def resolve_allegro_kuka_two_arms(cfg, *args, **kwargs):
    subtask_name: str = cfg["env"]["subtask"]
    subtask_map = dict(
        reorientation=AllegroKukaTwoArmsReorientation,
        regrasping=AllegroKukaTwoArmsRegrasping,
    )


    if subtask_name not in subtask_map:
        raise ValueError(f"Unknown subtask={subtask_name} in {subtask_map}")

    return subtask_map[subtask_name](cfg, *args, **kwargs)


# Mappings from strings to environments
isaacgym_task_map = {
    "FrankaCubeMove" : FrankaCubeMove,
    "FrankaTwoCubesMove" : FrankaTwoCubesMove
}
