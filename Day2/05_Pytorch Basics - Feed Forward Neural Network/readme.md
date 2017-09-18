## Python Programming in Elice

엘리스 플랫폼 용 Python 언어 예제입니다.
예제를 만들기 위해, 다음 파일들을 수정해주세요.

This is an example exercise for Elice platform using Python language.
Start from editing following files to make your own exercise.

<hr />

### Important files

#### readme.md

이 파일은 Markdown으로 쓰인 문서입니다.
이 파일의 내용은 웹페이지의 좌측에서 보입니다.

This file is a readme document written in Markdown language.
The content of this file will be shown on the left side of the webpage.

#### instructions.md

이 파일 역시 Markdown으로 쓰인 문서입니다.
이 파일의 내용은 웹페이지의 좌측 하단에 보여집니다.

This file contains detailed instructions for exercise tasks.
The contents will be shown at the bottom of the left side of the webpage.

#### .elice

.elice는 파일이 아닌, 디렉토리입니다.
이 디렉토리는 사용자로부터 감춰져 있으며, 예제를 실행하거나 채점하기 위한 코드를 담고있습니다.

This is not a file, but a directory.

This directory is hidden from users, and contains files to run and grade the exercise.

#### .elice/elice.conf

본 예제가 엘리스 플랫폼에서 실행되는 환경을 설정하는 파일입니다.
자세한 설명은 예제 파일에 입력되어있습니다.

This file contains configurations for running this exercise on Elice platform.
Detailed descriptions are written in the example file.

#### .elice/runner.sh

사용자가 예제페이지에서 실행 버튼을 클릭할 때 실행되는 shell script입니다.

웹페이지에서 user input은 `/dev/elice_in`에 저장되니, 코드 실행시 user input을 전달하시려면 `/dev/elice_in`을 사용하시면 됩니다.

스크립트 실행 시, 디렉토리 위치는 `{exercise_root}/.elice` 가 아닌 `{exercise_root}/` 인 점을 주의해주세요.

Elice platform will run this shell script when user clicks on the **Run** button on the exercise webpage.

User's input from the webpage will be available from `/dev/elice_in`,
so redirect this as an input to user's code when running.

Be aware that the current directory will be `{exercise_root}/` instead of `{exercise_root}/.elice` when this script is executed.

#### .elice/grader.sh

사용자가 예제 페이지에서 제출 버튼을 클릭할 때 실행되는 shell script 입니다.

제출된 코드를 채점하기 위해 `.elice` 디렉토리에 위치한 grader.py 를 실행합니다.

스크립트 실행 시, 디렉토리 위치는 `{exercise_root}/.elice` 가 아닌 `{exercise_root}/` 인 점을 주의해주세요.

Elice platform will run this shell script when user clicks on the **Submit** button on the exercise webpage.

Run your grader application in `.elice` directory to grade the submitted code.

Be aware that the current directory will be `{exercise_root}/` instead of `{exercise_root}/.elice` when this script is executed.

<hr />

#### Example: Images

엘리스에서 이미지를 삽입하려면 다음 포맷을 사용해주세요.

`![file_description]({{file_name}})`

벡터 이미지는 SVG 형식을 사용하면 더 깔끔하게 보입니다.

Inserting image in Elice is easy: `![file_description]({{file_name}})`.

For vector images and figures, we encourage to use `SVG` format for clean rendering.

![Raccoon]({{raccoon.jpg}})
