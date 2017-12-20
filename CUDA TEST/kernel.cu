#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "Camera.h"

GLFWwindow* windowInitialization();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
char* readglslFile(std::string);
int compileShader();

// window settings
const unsigned int SCR_WIDTH = 2000;
const unsigned int SCR_HEIGHT = 1600;
int cwidth = 2000, cheight = 1600;
const float deltaTimeFrame = 0.016f;


// Camera setting initialization
Camera camera = Camera(cwidth, cheight, glm::vec3(0, 5, 10), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
glm::vec2 mousePosition(0.f, 0.f);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);


const int xgrid = 80;
const int ygrid = 80;
const int zgrid = 40;
const int gridNumber = 256000;
const int particleNumber = 27648;
const int totalFrame = 10000;
__device__ const float deltaTime = 0.008f;

__device__ const float xmin = -3.f;
__device__ const float xmax = 5.f;
__device__ const float ymin = -3.f;
__device__ const float ymax = 5.f;
__device__ const float zmin = -3.f;
__device__ const float zmax = 1.f;

__device__ const float restDensity = 6378.f;
__device__ const float h = 0.1f;
__device__ const float k_corr = 0.0001f;
__device__ const float deltaq_corr = 0.03f;
__device__ const int n_corr = 4;
__device__ const float corrCoefficient = 4.86204e+16f;
__device__ const int solverIterations = 4;
__device__ const float epsilon = 600.f;
__device__ const float kernal1 = 1.5667e+09f;
__device__ const float kernal2 = 1.4324e+07f;
__device__ const float epsilonVorticity = 0.0005f;

class float33
{
public:
	float x, y, z;

	__device__ float33() : x(0.f), y(0.f), z(0.f) {}
	__device__ float33(float a, float b, float c) : x(a), y(b), z(c) {}

	__device__ float33 operator + (float33 vector);
	__device__ float33 operator - (float33 vector);
	__device__ float33 operator * (float number);
	__device__ float33 operator / (float number);

	__device__ float33 operator += (float33 vector);
	__device__ float33 operator -= (float33 vector);
	__device__ float33 operator *= (float number);
	__device__ float33 operator /= (float number);
};

__device__ float33 operator * (float number, float33 vector);
__device__ float float33Distance2(float33 vector);
__device__ float float33Distance(float33 vector);
__device__ float33 normalize(float33 vector);
__device__ float33 cross(float33 vector1, float33 vector2);

__device__ int intClamp(int number, int min, int max);
__device__ int intMax(int number1, int number2);
__device__ int intMin(int number1, int number2);
__device__ float floatMin(float number1, float number2);

__device__ float calculateDistance2(float x1, float x2, float y1, float y2, float z1, float z2)
{
	return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}

__device__ float calculateDistance2(float x, float y, float z)
{
	return x * x + y * y + z * z;
}

__global__ void addIntoGrid(float *position, float *newPosition, float *velocity, int *gridCounters, int *gridCells)
{
    int i = threadIdx.x + 1024 * blockIdx.x;
	velocity[3 * i + 1] -= deltaTime * 9.8f;
	newPosition[3 * i] = position[3 * i] + deltaTime * velocity[3 * i];
	newPosition[3 * i + 1] = position[3 * i + 1] + deltaTime * velocity[3 * i + 1];
	newPosition[3 * i + 2] = position[3 * i + 2] + deltaTime * velocity[3 * i + 2];

	int temp = intClamp(int((newPosition[3 * i] + 3) * 10), 0, xgrid - 1) * ygrid * zgrid
		     + intClamp(int((newPosition[3 * i + 1] + 3) * 10), 0, ygrid - 1) * zgrid
		     + intClamp(int((newPosition[3 * i + 2] + 3) * 10), 0, zgrid - 1);
	atomicAdd(&gridCells[temp * 16 + atomicAdd(&gridCounters[temp], 1)], i);
}

__global__ void calculateLambda(float *newPosition, int *gridCounters, int *gridCells ,float *lambda, float *test)
{
	int i = threadIdx.x + 1024 * blockIdx.x;
	float numerator = -restDensity, denominator = epsilon * restDensity * restDensity;
	float33 sumOfKernal = float33(0.f, 0.f, 0.f);
	float33 pi = float33(newPosition[3 * i], newPosition[3 * i + 1], newPosition[3 * i + 2]);
	int x = intClamp(int((pi.x + 3) * 10), 0, xgrid - 1),
		y = intClamp(int((pi.y + 3) * 10), 0, ygrid - 1),
		z = intClamp(int((pi.z + 3) * 10), 0, zgrid - 1);
	for (int ni = intMax(0, x - 1); ni <= intMin(xgrid - 1, x + 1); ++ni)
	{
		for (int nj = intMax(0, y - 1); nj <= intMin(ygrid - 1, y + 1); ++nj)
		{
			for (int nk = intMax(0, z - 1); nk <= intMin(zgrid - 1, z + 1); ++nk)
			{
				int temp = ni * ygrid * zgrid + nj * zgrid + nk;
				for (int j = 0; j < gridCounters[temp]; ++j)
				{
					int index = gridCells[temp * 16 + j];
					float33 pj = float33(newPosition[3 * index], newPosition[3 * index + 1], newPosition[3 * index + 2]);
					float distance2 = calculateDistance2(pi.x, pj.x, pi.y, pj.y, pi.z, pj.z);
					if (index != i && distance2 < h * h && distance2 > 0.000001f)
					{
						float33 spike = kernal2 * pow(h - sqrt(distance2), 2) * normalize(pi - pj);
						numerator += kernal1 * pow(h * h - distance2, 3);
						denominator += calculateDistance2(spike.x, spike.y, spike.z);
						sumOfKernal += spike;
					}
				}
			}
		}
	}
	denominator += calculateDistance2(sumOfKernal.x, sumOfKernal.y, sumOfKernal.z);
	lambda[i] = -floatMin(0.f, numerator) / denominator;
						test[i] = lambda[i];
}

__global__ void calculateDelta(float *newPosition, int *gridCounters, int *gridCells, float *lambda, float *delta)
{
	int i = threadIdx.x + 1024 * blockIdx.x;
	float33 sumDeltaP = float33(0.f, 0.f, 0.f);
	float33 pi = float33(newPosition[3 * i], newPosition[3 * i + 1], newPosition[3 * i + 2]);
	float lambdai = lambda[i];
	int x = intClamp(int((pi.x + 3) * 10), 0, xgrid - 1),
		y = intClamp(int((pi.y + 3) * 10), 0, ygrid - 1),
		z = intClamp(int((pi.z + 3) * 10), 0, zgrid - 1);
	for (int ni = intMax(0, x - 1); ni <= intMin(xgrid - 1, x + 1); ++ni)
	{
		for (int nj = intMax(0, y - 1); nj <= intMin(ygrid - 1, y + 1); ++nj)
		{
			for (int nk = intMax(0, z - 1); nk <= intMin(zgrid - 1, z + 1); ++nk)
			{
				int temp = ni * ygrid * zgrid + nj * zgrid + nk;
				for (int j = 0; j < gridCounters[temp]; ++j)
				{
					int index = gridCells[temp * 16 + j];
					float33 pj = float33(newPosition[3 * index], newPosition[3 * index + 1], newPosition[3 * index + 2]);
					float distance2 = calculateDistance2(pi.x, pj.x, pi.y, pj.y, pi.z, pj.z);
					if (index != i && distance2 < h * h && distance2 > 0.000001f)
					{
						sumDeltaP += (lambdai + lambda[index] - corrCoefficient * pow(h * h - distance2, 3 * n_corr)) * kernal2 * pow(h - sqrt(distance2), 2) * normalize(pi - pj);
					}
				}
			}
		}
	}
	delta[3 * i] = sumDeltaP.x;
	delta[3 * i + 1] = sumDeltaP.y;
	delta[3 * i + 2] = sumDeltaP.z;

	if (newPosition[3 * i] < xmin)
	{
		delta[3 * i] += xmin - newPosition[3 * i];
	}
	else if (newPosition[3 * i] > xmax)
	{
		delta[3 * i] += xmax - newPosition[3 * i];
	}
	if (newPosition[3 * i + 1] < ymin)
	{
		delta[3 * i + 1] += ymin - newPosition[3 * i + 1];
	}
	else if (newPosition[3 * i + 1] > ymax)
	{
		delta[3 * i + 1] += ymax - newPosition[3 * i + 1];
	}
	if (newPosition[3 * i + 2] < zmin)
	{
		delta[3 * i + 2] += zmin - newPosition[3 * i + 2];
	}
	else if (newPosition[3 * i + 2] > zmax)
	{
		delta[3 * i + 2] += zmax - newPosition[3 * i + 2];
	}
}

__global__ void calculatePredictedPosition(float *newPosition, float *delta)
{
	int i = threadIdx.x + 1024 * blockIdx.x;
	newPosition[3 * i] += delta[3 * i];
	newPosition[3 * i + 1] += delta[3 * i + 1];
	newPosition[3 * i + 2] += delta[3 * i + 2];
}

__global__ void calculateVelocityAndPosition(float *position, float *velocity, float *newPosition)
{
	int i = threadIdx.x + 1024 * blockIdx.x;
	velocity[3 * i] = (newPosition[3 * i] - position[3 * i]) / deltaTime;
	velocity[3 * i + 1] = (newPosition[3 * i + 1] - position[3 * i + 1]) / deltaTime;
	velocity[3 * i + 2] = (newPosition[3 * i + 2] - position[3 * i + 2]) / deltaTime;

	position[3 * i] = newPosition[3 * i];
	position[3 * i + 1] = newPosition[3 * i + 1];
	position[3 * i + 2] = newPosition[3 * i + 2];
}

void calculatePosition(float *position, float *velocity, float *lambda)
{
	float *dev_position = 0;
	float *dev_newPosition = 0;
	float *dev_velocity = 0;
	int *dev_gridCounters = 0;
	int *dev_gridCells = 0;
	float *dev_lambda = 0;
	float *dev_delta = 0;

	float *test = 0;

	cudaSetDevice(0);

	cudaMalloc((void**)&dev_position, particleNumber * 3 * sizeof(float));
	cudaMalloc((void**)&dev_newPosition, particleNumber * 3 * sizeof(float));
	cudaMalloc((void**)&dev_velocity, particleNumber * 3 * sizeof(float));
	cudaMalloc((void**)&dev_gridCounters, gridNumber * sizeof(int));
	//cudaMemset(dev_gridCounters, 0, gridNumber * sizeof(int));
	cudaMalloc((void**)&dev_gridCells, gridNumber * 16 * sizeof(int));
	//cudaMemset(dev_gridCells, 0, gridNumber * 16 * sizeof(int));
	cudaMalloc((void**)&dev_lambda, particleNumber * sizeof(float));
	cudaMalloc((void**)&dev_delta, particleNumber * 3 * sizeof(float));


	cudaMalloc((void**)&test, particleNumber * sizeof(float));
	cudaMemset(test, 0, particleNumber * sizeof(int));


	cudaMemcpy(dev_position, position, particleNumber * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_velocity, velocity, particleNumber * 3 * sizeof(float), cudaMemcpyHostToDevice);

	addIntoGrid<<<27, 1024>>>(dev_position, dev_newPosition, dev_velocity, dev_gridCounters, dev_gridCells);
	cudaDeviceSynchronize();

	for (int i = 0; i < solverIterations; ++i)
	{
		calculateLambda<<<27, 1024>>>(dev_newPosition, dev_gridCounters, dev_gridCells, dev_lambda, test);
		cudaDeviceSynchronize();

		calculateDelta<<<27, 1024>>>(dev_newPosition, dev_gridCounters, dev_gridCells, dev_lambda, dev_delta);
		cudaDeviceSynchronize();

		calculatePredictedPosition<<<27, 1024>>>(dev_newPosition, dev_delta);
		cudaDeviceSynchronize();
	}

	calculateVelocityAndPosition<<<27, 1024>>>(dev_position, dev_velocity, dev_newPosition);
	cudaDeviceSynchronize();

	cudaMemcpy(position, dev_position, particleNumber * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(velocity, dev_velocity, particleNumber * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(lambda, test, particleNumber * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_position);
	cudaFree(dev_newPosition);
	cudaFree(dev_velocity);
	cudaFree(dev_gridCounters);
	cudaFree(dev_gridCells);
	cudaFree(dev_lambda);
	cudaFree(dev_delta);
}

int main()
{
	// Initialize a window
	GLFWwindow* window = windowInitialization();

	// Read shader file and compile shader program
	int shaderProgram = compileShader();
	glUseProgram(shaderProgram);

	// Opengl Settings
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	// Initialize vertex buffer
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	// Initialize depth buffer
	GLuint mdepthRenderBuffer = -1;
	glBindRenderbuffer(GL_RENDERBUFFER, mdepthRenderBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, cwidth, cheight);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mdepthRenderBuffer);


	// Initialize particle value, index and position for openGL
	float *position = new float[particleNumber * 3];
	float *velocity = new float[particleNumber * 3]();
	int *indices = new int[particleNumber];
	float *lambda = new float[particleNumber];

	int number = 0;
	for (int i = -24; i < 0; ++i)
	{
		for (int j = -24; j < 24; ++j)
		{
			for (int k = -24; k < 0; ++k)
			{
				position[number++] = 0.11f * i + (rand() % 10000) / 1000000.0f;
				position[number++] = 0.11f * j + (rand() % 10000) / 1000000.0f;
				position[number++] = 0.11f * k + (rand() % 10000) / 1000000.0f;
			}
		}
	}

	for (int i = 0; i < particleNumber; i++)
	{
		indices[i] = i;
	}

	int attrPos = glGetAttribLocation(shaderProgram, "Position");
	int unifProjView = glGetUniformLocation(shaderProgram, "ProjViewMatrix");
	int unifView = glGetUniformLocation(shaderProgram, "ViewMatrix");

	// set up vertex data and configure vertex attributes
	unsigned int VBO, EBO;

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, particleNumber * 3 * sizeof(float), position, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, particleNumber * sizeof(int), indices, GL_STATIC_DRAW);


	// render loop
	double previousTime = glfwGetTime();
	int currentFrame = 0;
	double accumulateTime = 0;

	while (!glfwWindowShouldClose(window))
	{
		// time calculation
		accumulateTime += glfwGetTime() - previousTime;
		previousTime = glfwGetTime();

		if (accumulateTime >= deltaTimeFrame)
		{
			accumulateTime = accumulateTime - deltaTimeFrame;
			if (currentFrame < totalFrame)
			{
				calculatePosition(position, velocity, lambda);
				
				int N = 100;
				for (int j = 1; j < particleNumber / N; ++j)
				{
					if (lambda[j * N] > 0 || lambda[j * N] < 0)
					std::cout << lambda[j * N] << ' ';
				}
				std::cout << std::endl;
			}
			currentFrame++;
			if (currentFrame == totalFrame)
			{
				currentFrame = 0;
			}

			// input
			processInput(window);

			// render
			glClearColor(0.8f, 0.9f, 1.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, particleNumber * 3 * sizeof(float), position, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(attrPos);
			glVertexAttribPointer(attrPos, 3, GL_FLOAT, false, 0, NULL);

			glm::mat4 ProjViewMatrix = camera.getProj() * camera.getView();
			glm::mat4 ViewMatrix = camera.getView();

			glUniformMatrix4fv(unifProjView, 1, GL_FALSE, &ProjViewMatrix[0][0]);
			glUniformMatrix4fv(unifView, 1, GL_FALSE, &ViewMatrix[0][0]);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glDrawElements(GL_POINTS, particleNumber, GL_UNSIGNED_INT, 0);


			double xpos, ypos;
			glfwGetCursorPos(window, &xpos, &ypos);
			glm::vec2 pos(xpos, ypos);
			if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
			{
				glm::vec2 diff = 0.05f * (pos - mousePosition);
				camera.RotatePhi(-diff.x);
				camera.RotateTheta(-diff.y);
			}
			else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
			{
				glm::vec2 diff = 0.005f * (pos - mousePosition);
				camera.TranslateAlongRight(-diff.x);
				camera.TranslateAlongUp(diff.y);
			}
			mousePosition = pos;

			glfwSetScrollCallback(window, scroll_callback);

			// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	}
	// de-allocate all resources once they've outlived their purpose:
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	glfwTerminate();

	/*
	std::ofstream fout;
	fout.open("test.txt");
	fout << particleNumber << " " << totalFrame / 5 << " ";



		if (i / 5 == 0)
		{
			for (int j = 0; j < particleNumber; ++j)
			{
				fout << position[3 * j] << " " << position[3 * j + 1] << " " << position[3 * j + 2] << " ";
			}
		}

	fout.close();

	
	for (int i = 1; i <= totalFrame; ++i)
	{
		std::cout << "Current frame is " << i << std::endl;
		calculatePosition(position, velocity);
	}
	*/

    cudaDeviceReset();
	return 0;
}


__device__ float33 float33::operator + (float33 vector)
{
	return float33(x + vector.x, y + vector.y, z + vector.z);
}

__device__ float33 float33::operator - (float33 vector)
{
	return float33(x - vector.x, y - vector.y, z - vector.z);
}

__device__ float33 float33::operator * (float number)
{
	return float33(x * number, y * number, z * number);
}

__device__ float33 float33::operator / (float number)
{
	return float33(x / number, y / number, z / number);
}

__device__ float33 float33::operator += (float33 vector)
{
	return *this + vector;
}

__device__ float33 float33::operator -= (float33 vector)
{
	return *this - vector;
}

__device__ float33 float33::operator *= (float number)
{
	return *this * number;
}

__device__ float33 float33::operator /= (float number)
{
	return *this / number;
}

__device__ float33 operator * (float number, float33 vector)
{
	return vector * number;
}

__device__ float float33Distance2(float33 vector)
{
	return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
}

__device__ float float33Distance(float33 vector)
{
	return sqrt(float33Distance2(vector));
}

__device__ float33 normalize(float33 vector)
{
	return vector / float33Distance(vector);
}

__device__ float33 cross(float33 vector1, float33 vector2)
{
	return float33(vector1.y * vector2.z - vector1.z * vector2.y, vector1.z * vector2.x - vector1.x * vector2.z, vector1.x * vector2.y - vector1.y * vector2.x);
}

__device__ int intMax(int number1, int number2)
{
	if (number1 > number2) return number1;
	return number2;
}

__device__ int intMin(int number1, int number2)
{
	if (number1 < number2) return number1;
	return number2;
}

__device__ float floatMin(float number1, float number2)
{
	if (number1 < number2) return number1;
	return number2;
}

__device__ int intClamp(int number, int min, int max)
{
	if (number <= min) return min;
	if (number >= max) return max;
	return number;
}



// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	cwidth = width;
	cheight = height;
	glViewport(0, 0, width, height);
}

GLFWwindow* windowInitialization()
{
	// glfw: initialize and configure
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// glfw window creation
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Fluid", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// glad: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
	}
	return window;
}

char* readglslFile(std::string filename)
{
	int array_size = 10000;
	char *shaderSource = new char[array_size];
	int pos = 0;
	std::ifstream file(filename);
	if (file.is_open())
	{
		while (!file.eof() && pos < array_size)
		{
			shaderSource[pos] = file.get();
			pos++;
		}
		shaderSource[pos - 1] = '\0';
	}
	file.close();
	return shaderSource;
}

int compileShader()
{
	char *vertexShaderSource = readglslFile("particle.vert.txt");
	char *fragmentShaderSource = readglslFile("particle.frag.txt");

	// build and compile shader program
	// vertex shader
	int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	// check for shader compile errors
	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// fragment shader
	int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	// check for shader compile errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// link shaders
	int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	// check for linking errors
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.Zoom(yoffset * 0.1);
}