Lab 1 code layout

This submission keeps the runnable code files at the top level of the code/ folder.
Older per-person working directories are preserved under code/other/ for reference.

Top-level runnable files

- code/1c.py: Question 1(c)
- code/2a.py: Question 2(a)
- code/2b.py: Question 2(b)
- code/benchmark_3ab.py: Questions 3(a) and 3(b)
- code/decoder.py and code/optimizer.py: helper modules used by benchmark_3ab.py
- code/part4.py: Questions 4(a) and 4(b)
- code/part4_debug_and_4c.py: Question 4(c)
- code/part1c.py, code/part2.py, code/read_data.py, code/ref_optimize.py: helper files used by the Question 4 code
- code/5.py: Question 5
- code/results/: Question 5 plots and summary used in the report

Reference archive

- code/other/angelo/: original per-person folder preserved for reference
- code/other/shah/: original per-person folder preserved for reference
- code/other/khoi/: original per-person folder preserved for reference

Submission outputs

The four required result files are stored in the top-level result/ folder:

- result/decode_output.txt
- result/gradient.txt
- result/solution.txt
- result/prediction.txt

Additional report artifacts are stored separately:

- result/other/benchmark/: Question 3 benchmark plots and metrics
- result/other/: extra Question 4 output files
- code/results/: Question 5 plots and summary
- code/other/khoi/generated_graphs/: Question 4 figures used in the report

How to run

Run all commands from the repository root.

Question 1(c):
- python code/1c.py

Question 2(a):
- python code/2a.py

Question 2(b):
- python code/2b.py

Question 3(a) and 3(b):
- python code/benchmark_3ab.py

Question 4(a) and 4(b):
- python code/part4.py

Question 4(c):
- python code/part4_debug_and_4c.py

Question 5:
- python code/5.py

Dependencies

- Core code uses Python with NumPy and SciPy.
- Question 5 also uses matplotlib, OpenCV, and liblinear.
