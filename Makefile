-include ./testing_recipes/makefiles/Makefile

setup-submodule:
	@$(MAKE) $(MAKE_TAG) init-submodule
	@$(MAKE) $(MAKE_TAG) update-submodule
.PHONY: setup-testing_recipes


test-all:  ## Run all tests
	# @$(MAKE) ${MAKE_TAG} echo-cyan msg="2.1. Unit tests"
	# @$(MAKE) ${MAKE_TAG} pytest mark=ut
	@$(MAKE) ${MAKE_TAG} echo-cyan msg="2.2. Test notebooks"
	@$(MAKE) ${MAKE_TAG} pytest mark=tuto
.PHONY: test-all
