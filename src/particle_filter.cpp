/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	default_random_engine gen;

  // Set the number of particles
	num_particles = 300;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

  // Create normal (Gaussian) distributions for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(particle.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	default_random_engine gen;
	double x_f, y_f, theta_f;

	for (auto &particle : particles) {
		if (fabs(yaw_rate) < 1e-6) {
			// Motion model (bicycle model) when yaw_rate is 0
			x_f = particle.x + velocity * delta_t * cos(particle.theta);
			y_f = particle.y + velocity * delta_t * sin(particle.theta);
			theta_f = particle.theta;
		} else {
			// Motion model (bicycle model) when yaw_rate is not 0
			x_f = particle.x + velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			y_f = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
			theta_f = particle.theta +  yaw_rate * delta_t;
		}

		double std_x = std_pos[0];
		double std_y = std_pos[1];
		double std_theta = std_pos[2];

		normal_distribution<double> dist_x_f(x_f, std_x);
		normal_distribution<double> dist_y_f(y_f, std_y);
		normal_distribution<double> dist_theta_f(theta_f, std_theta);

		particle.x = dist_x_f(gen);
		particle.y = dist_y_f(gen);
		particle.theta = dist_theta_f(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	for (auto &observation : observations) {
		double dist_min = 1000.0;
		for (auto &landmark : predicted) {
			double distance = dist(observation.x, observation.y, landmark.x, landmark.y);
			if (distance < dist_min) {
				dist_min = distance;
				observation.id = landmark.id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution.
	for (int i = 0; i < num_particles; i++) {
		Particle& particle = particles[i];
		particle.weight = 1.0;
		double x_p = particle.x;
		double y_p = particle.y;
		double theta_p = particle.theta;
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

    vector<LandmarkObs> observations_m; // Observations in MAP'S coordinate system
		for (auto &observation : observations) {
			LandmarkObs observation_m;
			double x_c = observation.x;
			double y_c = observation.y;

      // Assign id for observation_m to -1 since it's not associated with any landmark yet
      observation_m.id = -1;
			// Coordinate transform (rotation and translation)
			// 	 Observations - VEHICLE'S coordinate system
			//   Particles - MAP'S coordinate system
			observation_m.x = x_p + cos(theta_p) * x_c - sin(theta_p) * y_c;
			observation_m.y = y_p + sin(theta_p) * x_c + cos(theta_p) * y_c;
			observations_m.push_back(observation_m);
		}

		vector<LandmarkObs> predicted;
		for (auto &map_landmark : map_landmarks.landmark_list) {
			double dist_landmark = dist(x_p, y_p, map_landmark.x_f, map_landmark.y_f);
			if (dist_landmark < sensor_range) {
				// Find all landmarks within the sensor range
				LandmarkObs landmark;
				landmark.id = map_landmark.id_i;
				landmark.x = map_landmark.x_f;
				landmark.y = map_landmark.y_f;
				predicted.push_back(landmark);
			}
		}

		dataAssociation(predicted, observations_m);

    double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		bool no_valid_associations = true;
		for (auto &observation : observations_m) {
			if (observation.id != -1) {
				double x_obs = observation.x;
				double y_obs = observation.y;

				for (auto &map_landmark : map_landmarks.landmark_list) {
					if (map_landmark.id_i == observation.id) {
						double mu_x = map_landmark.x_f;
					  double mu_y = map_landmark.y_f;
						// Calculate normalization term
						double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));
						// Calculate exponent
						double exponent= pow((x_obs - mu_x),2.0)/(2 * pow(sig_x,2.0)) + pow((y_obs - mu_y),2.0)/(2 * pow(sig_y,2.0));
						// Calculate weight using normalization terms and exponent
						double weight = gauss_norm * exp(-exponent);
						particle.weight *= weight;
						sense_x.push_back(x_obs);
						sense_y.push_back(y_obs);
						associations.push_back(map_landmark.id_i);
						no_valid_associations = false;
					}
				}
				if (no_valid_associations) {
					particle.weight = 0.0;
				}
				particle = SetAssociations(particle, associations, sense_x, sense_y);
				weights[i] = particle.weight;
			}
		}
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<double> distribution(weights.begin(), weights.end());

	vector<Particle> particles_new;
	for (int i = 0; i < num_particles; i++) {
		particles_new.push_back(particles[distribution(gen)]);
	}
	particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

		return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
