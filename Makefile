MKDIR_P = mkdir -p
BUILD_DIR = build/
XCODE_DIR = xcode/

.PHONY: dirs xcode-project

all: dirs ${BUILD_DIR}/raymarcher

${BUILD_DIR} ${XCODE_DIR}:
	${MKDIR_P} $@

${BUILD_DIR}/raymarcher: ${BUILD_DIR}
	cd ${BUILD_DIR}; cmake ../renderer/; make

xcode-project: ${XCODE_DIR}
	cd ${XCODE_DIR}; cmake -G Xcode ../renderer/

clean:
	-@rm -r ${BUILD_DIR} ||:
	-@rm -r ${XCODE_DIR} ||:
