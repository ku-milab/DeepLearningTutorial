import os, sys

sys.path.append(os.getcwd())
import elice_utils


def answer_func(x, y):
    return x + y

def grade():
    import main as submission
    total_score = 0

    #### TEST CASES ####
    testcases = [(3, 5), (10, 20), (473, 1947)]
    scores = [30, 30, 40]

    for testcase_index in range(len(testcases)):
        testcase = testcases[testcase_index]
        current_prob_score = scores[testcase_index]

        student_answer = submission.simple_task(testcase[0], testcase[1])
        correct_answer = answer_func(testcase[0], testcase[1])

        # COMPARE TWO SOLUTIONS
        if student_answer == correct_answer:
            # IF CORRECT, ADD SCORE
            total_score += current_prob_score
            elice_utils.secure_send_grader('Testcase %d: accept (%d points)\n' % (testcase_index + 1, current_prob_score))
        else:
            elice_utils.secure_send_grader('Testcase %d: wrong\n' % (testcase_index + 1))

    elice_utils.secure_send_image('raccoon.jpg')
    elice_utils.secure_send_file('main.py')
    elice_utils.secure_send_grader('\nTotal score: %d points\n' % (total_score))
    # SEND SCORE TO ELICE
    elice_utils.secure_send_score(total_score)

try:
    elice_utils.secure_init()
    grade()
except Exception as err:
    elice_utils.secure_send_grader('An exception occurred testing your implementation. Please run and check that it works before submitting.\n')
    elice_utils.secure_send_grader('Error info: ' + str(err))
    elice_utils.secure_send_score(0)
    sys.exit(1)
