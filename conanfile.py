from conan import ConanFile
from conan.tools.cmake import cmake_layout


class ExampleRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("opencv/4.11.0")
        self.requires("onnxruntime/1.15.1")
        self.requires("classifier_inference/1.0.0")
        self.requires("detector_inference/1.0.0")
        self.requires("sequencer_inference/1.0.0")
        

    def layout(self):
        cmake_layout(self)