SCENES_DIR=/home/isaiah/scisim/assets/rigidbody2d/isaiah_ura/scenes/
SIM_OUT_DIR=/home/isaiah/scisim/assets/rigidbody2d/isaiah_ura/outs/
SUB_DIR=general/

.PHONY: test
test:
	python3 -m pytest .

.PHONY: generate_scenes
generate_scenes:
	python3 generate_balls.py $(SCENES_DIR)

.PHONY: run_sim_ok
run_sim_ok:
	SUB_DIR=ok/ make run_sim

.PHONY: run_sim_general
run_sim_ok:
	SUB_DIR=general/ make run_sim

.PHONY: run_sim
run_sim:
	python3 runner.py $(SCENES_DIR)$(SUB_DIR) $(SIM_OUT_DIR)$(SUB_DIR)

.PHONY: gen_and_run
gen_and_run: generate_scenes run_sim