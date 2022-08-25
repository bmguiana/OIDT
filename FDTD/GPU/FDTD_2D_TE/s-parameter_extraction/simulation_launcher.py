"""
Author: B. Guiana

Description:


Acknowledgement: This project was completed as part of research conducted with
                 my major professor and advisor, Prof. Ata Zadehgol, at the
                 Applied and Computational Electromagnetics Signal and Power
                 Integrity (ACEM-SPI) Lab while working toward the Ph.D. in
                 Electrical Engineering at the University of Idaho, Moscow,
                 Idaho, USA. This project was funded, in part, by the National
                 Science Foundation (NSF); award #1816542 [1].

"""

import user_config as usr
import os
from time import sleep
from subprocess import run

if usr.sim_type == 's-param':
    print('launching S-paramter Extraction Simulation')
    print('Simulating {} coupled lines ({} sims total)'.format(usr.num_lines, usr.num_lines*4))

    for line in range(usr.num_lines):
        print('Evaluating line {}'.format(line+1))
        run('python fdtd_main.py {} f i'.format(line), shell=True)
        sleep(5)
        run('python fdtd_main.py {} f r'.format(line), shell=True)
        sleep(5)
        run('python fdtd_main.py {} b i'.format(line), shell=True)
        sleep(5)
        run('python fdtd_main.py {} b r'.format(line), shell=True)
        sleep(5)

    print('\n\n\nEvaluating S-Parameters')
    run('python extract_sparams.py', shell=True)
    print('Done extracting S-parameters! The touchstone file is in {}'.format(usr.output_dir))
    print('\n\n\n')
    print('To continue with equivalent circuit synthesis and passivity enforcement:')
    print('1. Copy {}.s{}p from the folder ./{} to the folder SROPEE/Input'.format(usr.sparam_file, usr.num_lines*2, usr.output_dir))
    print('2. Follow the directions in SROPEE/Instructions.pdf for further instructions on continuing with equivalent circuit synthesis and passivity enforcement')
    file = open('./{}/NEXT_STEPS_INSTRUCTIONS.txt'.format(usr.output_dir), mode='w')
    file_lines = ['To continue with equivalent circuit synthesis and passivity enforcement:\n1. Copy {}.s{}p from the folder ./{} to the folder SROPEE/Input\n2. Follow the directions in SROPEE/Instructions.pdf for further instructions on continuing with equivalent circuit synthesis and passivity enforcement'.format(usr.sparam_file, usr.num_lines*2, usr.output_dir)]
    file.writelines(file_lines)
    file.close()

else:
    print('Test Environment')
