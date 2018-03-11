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

const double INIT_PARTICLE_WEIGHT = 1.0;
const double MIN_YAW = 1e-6;
using namespace std;

// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
//   x, y, theta and their uncertainties from GPS) and all weights to 1.
// Add random Gaussian noise to each particle.
// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
void ParticleFilter::init(double gps_x, double gps_y, double theta, double std[]) {
  num_particles = 100;
  particles.resize(num_particles);
  weights.resize(num_particles);

  default_random_engine gen;
  normal_distribution<double> dist_x(gps_x, std[0]);
  normal_distribution<double> dist_y(gps_y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = INIT_PARTICLE_WEIGHT;
    particles[i] = p;
    weights[i] = p.weight;
  }

  is_initialized = true;
}

// TODO: Add measurements to each particle and add random Gaussian noise.
// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
//  http://www.cplusplus.com/reference/random/default_random_engine/
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  double vel_by_yaw = velocity / yaw_rate;

  for(int i = 0; i < num_particles; ++i) {
    Particle *p = &particles[i];

    if(fabs(yaw_rate) < MIN_YAW) {
      p->x += velocity * delta_t * cos(p->theta);
      p->y += velocity * delta_t * sin(p->theta);
    } else {
      double theta = p->theta + yaw_rate * delta_t; // updated theta
      p->x += vel_by_yaw * (sin(theta) - sin(p->theta));
      p->y += vel_by_yaw * (cos(p->theta) - cos(theta));
      p->theta = theta;
    }

    p->x += dist_x(gen);
    p->y += dist_y(gen);
    p->theta += dist_theta(gen);
  }
}

// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
//   observed measurement to this particular landmark.
// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
//   implement this method and use it as a helper during the updateWeights phase.
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks_visible_to_particle,
                                     std::vector<LandmarkObs>& trans_observations) {
  //  cout << endl << endl;

  for(unsigned long i = 0; i < trans_observations.size(); ++i) {

    double current_min = numeric_limits<double>::max();
    unsigned int min_index = -1;

    for(unsigned long j = 0; j < landmarks_visible_to_particle.size(); ++j) {

      double current_dist = dist(trans_observations[i].x,
                                 trans_observations[i].y,
                                 landmarks_visible_to_particle[j].x,
                                 landmarks_visible_to_particle[j].y);

      //cout << "landmark, j, dist = " << landmarks_visible_to_particle[j].x << "," << landmarks_visible_to_particle[j].y << "," << j << "," << current_dist << endl;

      if(current_dist < current_min) {
        current_min = current_dist;
        min_index = j;
      }
    }

    trans_observations[i].id = min_index;
    //    cout << "observation and nearest = " << trans_observations[i].x << "," << trans_observations[i].y << endl;
    //    cout << "landmark, j = " << landmarks_visible_to_particle[min_index].x << "," << landmarks_visible_to_particle[min_index].y << ", "<< min_index << endl;
  }

  //  cout << endl << endl;
}

// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
//   The following is a good resource for the theory:
//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//   and the following is a good resource for the actual equation to implement (look at equation
//   3.33
//   http://planning.cs.uiuc.edu/node99.html
void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  // 1. Convert observations from Car Co-ordinates to Map Co-ordinates.
  std::vector<LandmarkObs> trans_observations;
  std::vector<LandmarkObs> landmarks_visible_to_particle;
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double gaussian_norm = (1 / (2 * M_PI * std_x * std_y));
  double sum_of_weights = 0.0;

  for(unsigned int i = 0; i < particles.size(); ++i) {
    Particle *p = &particles[i];
    double weight = 1.0;

    for(unsigned int j = 0; j < observations.size(); ++j) {
      LandmarkObs obs;
      LandmarkObs trans_obs;

      trans_obs.x = p->x + (obs.x * cos(p->theta)) - (obs.y * sin(p->theta));
      trans_obs.y = p->y + (obs.x * sin(p->theta)) + (obs.y * cos(p->theta));
      trans_obs.id = obs.id;

      Map::single_landmark_s closest_lm;
      double dist_min = numeric_limits<double>::max();

      for(unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
        Map::single_landmark_s current_lm = map_landmarks.landmark_list[k];
        double _dist = dist(trans_obs.x, trans_obs.y, current_lm.x_f, current_lm.y_f);
        if(_dist < dist_min) {
          dist_min = _dist;
          closest_lm = current_lm;
        }
      }

      // Now using closest lm and trans_obs, update weights using
      // Multivariate Gaussian Distribution
      double x = pow((trans_obs.x - closest_lm.x_f), 2) / (2 * pow(std_x, 2));
      double y = pow((trans_obs.y - closest_lm.y_f), 2) / (2 * pow(std_y, 2));
      double exponent = exp(-(x + y));
      weight *= gaussian_norm * exponent;
    } // for each observation

    sum_of_weights += weight;
    p->weight = weight;
  } // For each particle

  for(int i = 0; i < num_particles; ++i) {
    Particle *p = &particles[i];
    p->weight /= sum_of_weights;
    weights[i] = p->weight;
  }
}

// TODO: Resample particles with replacement with probability proportional to their weight.
// NOTE: You may find std::discrete_distribution helpful here.
//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
void ParticleFilter::resample() {
  vector<Particle> resampled_particles;
  default_random_engine gen;
  discrete_distribution<int> index(weights.begin(), weights.end());

  for(unsigned int i = 0; i < particles.size(); ++i) {
    int j = index(gen);
    Particle p {
      j,
      particles[j].x,
      particles[j].y,
      particles[i].theta,
      1
    };

    resampled_particles.push_back(p);
  }

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
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
