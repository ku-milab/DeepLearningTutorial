import os, sys; sys.path.append(os.getcwd())
import elice_utils
elice_utils.secure_init()

try:
    total_score = 0

    elice_utils.secure_send_grader('Grading...\n')
    elice_utils.secure_send_score(total_score)
except Exception as e:
    elice_utils.secure_send_grader('Internal Error:\n%s\n' % str(err))
    elice_utils.secure_send_score(0)
    sys.exit(1)
