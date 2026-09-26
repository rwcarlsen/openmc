#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openmc/cell.h"
#include "openmc/surface.h"

#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>

#include <pugixml.hpp>

namespace {

// Helper class to set up and tear down test surfaces
class SurfaceFixture {
public:
  SurfaceFixture()
  {
    pugi::xml_document doc;
    pugi::xml_node surf_node = doc.append_child("surface");
    surf_node.set_name("surface");
    surf_node.append_attribute("id") = "0";
    surf_node.append_attribute("type") = "x-plane";
    surf_node.append_attribute("coeffs") = "1";

    for (int i = 1; i < 10; ++i) {
      surf_node.attribute("id") = i;
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceXPlane>(surf_node));
      openmc::model::surface_map[i] = i - 1;
    }
  }

  ~SurfaceFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }
};

// Helper class for testing multiple intersections with the same surface
class MultiIntersectionFixture {
public:
  MultiIntersectionFixture()
  {
    pugi::xml_document doc;

    auto plane = doc.append_child("surface");
    plane.append_attribute("id") = 1;
    plane.append_attribute("type") = "x-plane";
    plane.append_attribute("coeffs") = "5";
    openmc::model::surfaces.push_back(
      std::make_unique<openmc::SurfaceXPlane>(plane));
    openmc::model::surface_map[1] = 0;

    auto sphere = doc.append_child("surface");
    sphere.append_attribute("id") = 2;
    sphere.append_attribute("type") = "sphere";
    sphere.append_attribute("coeffs") = "5 0 0 1";
    openmc::model::surfaces.push_back(
      std::make_unique<openmc::SurfaceSphere>(sphere));
    openmc::model::surface_map[2] = 1;
  }

  ~MultiIntersectionFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }
};

// Helper class for testing coincident surfaces with different representations
class CoincidentSurfaceFixture {
public:
  CoincidentSurfaceFixture()
  {
    pugi::xml_document doc;

    auto cylinder = doc.append_child("surface");
    cylinder.append_attribute("id") = 1;
    cylinder.append_attribute("type") = "z-cylinder";
    cylinder.append_attribute("coeffs") = "0 0 499.0000001";
    openmc::model::surfaces.push_back(
      std::make_unique<openmc::SurfaceZCylinder>(cylinder));
    openmc::model::surface_map[1] = 0;

    auto quadric = doc.append_child("surface");
    quadric.append_attribute("id") = 2;
    quadric.append_attribute("type") = "quadric";
    quadric.append_attribute("coeffs") =
      "1 1 7.498798913309288e-33 -7.498798913309288e-33 "
      "-1.2246467991473532e-16 -1.2246467991473532e-16 0 0 0 "
      "-249001.00009980003";
    openmc::model::surfaces.push_back(
      std::make_unique<openmc::SurfaceQuadric>(quadric));
    openmc::model::surface_map[2] = 1;
  }

  ~CoincidentSurfaceFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }
};

class CountingXPlane : public openmc::SurfaceXPlane {
public:
  CountingXPlane(pugi::xml_node node, int& distance_calls, int& sense_calls)
    : SurfaceXPlane {node}, distance_calls_ {distance_calls},
      sense_calls_ {sense_calls}
  {}

  double evaluate(openmc::Position r) const override
  {
    ++sense_calls_;
    return SurfaceXPlane::evaluate(r);
  }

  double distance(openmc::Position r, openmc::Direction u,
    bool coincident) const override
  {
    ++distance_calls_;
    return SurfaceXPlane::distance(r, u, coincident);
  }

private:
  int& distance_calls_;
  int& sense_calls_;
};

template<class Plane>
class CountingKernelPlane : public Plane {
public:
  CountingKernelPlane(
    pugi::xml_node node, int& distance_calls, int& normal_calls, int& sense_calls)
    : Plane {node}, distance_calls_ {distance_calls},
      normal_calls_ {normal_calls}, sense_calls_ {sense_calls}
  {}

  double evaluate(openmc::Position r) const override
  {
    ++sense_calls_;
    return Plane::evaluate(r);
  }

  double distance(openmc::Position r, openmc::Direction u,
    bool coincident) const override
  {
    ++distance_calls_;
    return Plane::distance(r, u, coincident);
  }

  openmc::Direction normal(openmc::Position r) const override
  {
    ++normal_calls_;
    return Plane::normal(r);
  }

private:
  int& distance_calls_;
  int& normal_calls_;
  int& sense_calls_;
};

class PlaneKernelFixture {
public:
  PlaneKernelFixture(const char* type, const char* coefficients)
  {
    pugi::xml_document doc;
    auto plane = doc.append_child("surface");
    plane.append_attribute("id") = 1;
    plane.append_attribute("type") = type;
    plane.append_attribute("coeffs") = coefficients;
    const std::string plane_type {type};
    if (plane_type == "x-plane") {
      openmc::model::surfaces.push_back(
        std::make_unique<CountingKernelPlane<openmc::SurfaceXPlane>>(
          plane, distance_calls, normal_calls, sense_calls));
    } else if (plane_type == "y-plane") {
      openmc::model::surfaces.push_back(
        std::make_unique<CountingKernelPlane<openmc::SurfaceYPlane>>(
          plane, distance_calls, normal_calls, sense_calls));
    } else if (plane_type == "z-plane") {
      openmc::model::surfaces.push_back(
        std::make_unique<CountingKernelPlane<openmc::SurfaceZPlane>>(
          plane, distance_calls, normal_calls, sense_calls));
    } else {
      openmc::model::surfaces.push_back(
        std::make_unique<CountingKernelPlane<openmc::SurfacePlane>>(
          plane, distance_calls, normal_calls, sense_calls));
    }
    openmc::model::surface_map[1] = 0;
  }

  ~PlaneKernelFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }

  int distance_calls {0};
  int normal_calls {0};
  int sense_calls {0};
};

class CountingPlaneFixture {
public:
  CountingPlaneFixture()
  {
    for (int i = 0; i < 3; ++i) {
      pugi::xml_document doc;
      auto node = doc.append_child("surface");
      node.append_attribute("id") = i + 1;
      node.append_attribute("type") = "x-plane";
      node.append_attribute("coeffs") = i + 1;
      openmc::model::surfaces.push_back(std::make_unique<CountingXPlane>(
        node, distance_calls[i], sense_calls[i]));
      openmc::model::surface_map[i + 1] = i;
    }
  }

  ~CountingPlaneFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }

  std::array<int, 3> distance_calls {};
  std::array<int, 3> sense_calls {};
};

class CountingSphere : public openmc::SurfaceSphere {
public:
  CountingSphere(pugi::xml_node node, int& distance_calls, int& sense_calls)
    : SurfaceSphere {node}, distance_calls_ {distance_calls},
      sense_calls_ {sense_calls}
  {}

  double evaluate(openmc::Position r) const override
  {
    ++sense_calls_;
    return SurfaceSphere::evaluate(r);
  }

  double distance(openmc::Position r, openmc::Direction u,
    bool coincident) const override
  {
    ++distance_calls_;
    return SurfaceSphere::distance(r, u, coincident);
  }

private:
  int& distance_calls_;
  int& sense_calls_;
};

class UnboundedCountingSphere : public CountingSphere {
public:
  using CountingSphere::CountingSphere;

  openmc::BoundingBox bounding_box(bool) const override
  {
    return openmc::BoundingBox::infinite();
  }
};

class MisboundedCountingSphere : public CountingSphere {
public:
  using CountingSphere::CountingSphere;

  openmc::BoundingBox bounding_box(bool) const override
  {
    return {{10.0, 10.0, 10.0}, {11.0, 11.0, 11.0}};
  }
};

class ScaledPlaneFixture {
public:
  ScaledPlaneFixture()
  {
    pugi::xml_document doc;
    auto plane = doc.append_child("surface");
    plane.append_attribute("id") = 1;
    plane.append_attribute("type") = "plane";
    plane.append_attribute("coeffs") = "1000000 0 0 0";
    openmc::model::surfaces.push_back(
      std::make_unique<openmc::SurfacePlane>(plane));
    openmc::model::surface_map[1] = 0;
  }

  ~ScaledPlaneFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }
};

class CurvedEventFixture {
public:
  explicit CurvedEventFixture(bool coincident_spheres = false)
  {
    pugi::xml_document doc;
    auto sphere = doc.append_child("surface");
    sphere.append_attribute("id") = 1;
    sphere.append_attribute("type") = "sphere";
    sphere.append_attribute("coeffs") =
      coincident_spheres ? "0 0 0 1" : "5 0 0 1";
    openmc::model::surfaces.push_back(std::make_unique<CountingSphere>(
      sphere, distance_calls[0], sense_calls[0]));
    openmc::model::surface_map[1] = 0;

    auto second = doc.append_child("surface");
    second.append_attribute("id") = 2;
    if (coincident_spheres) {
      second.append_attribute("type") = "sphere";
      second.append_attribute("coeffs") = "0 0 0 1";
      openmc::model::surfaces.push_back(std::make_unique<CountingSphere>(
        second, distance_calls[1], sense_calls[1]));
    } else {
      second.append_attribute("type") = "x-plane";
      second.append_attribute("coeffs") = 5;
      openmc::model::surfaces.push_back(std::make_unique<CountingXPlane>(
        second, distance_calls[1], sense_calls[1]));
    }
    openmc::model::surface_map[2] = 1;
  }

  ~CurvedEventFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }

  std::array<int, 2> distance_calls {};
  std::array<int, 2> sense_calls {};
};

class BoundedBranchFixture {
public:
  static constexpr int N_SURFACES {12};

  explicit BoundedBranchFixture(bool bounded = true)
  {
    for (int i = 0; i < N_SURFACES; ++i) {
      pugi::xml_document doc;
      auto sphere = doc.append_child("surface");
      sphere.append_attribute("id") = i + 1;
      sphere.append_attribute("type") = "sphere";
      const std::string coefficients =
        "0 " + std::to_string(4 * i) + " 0 1";
      sphere.append_attribute("coeffs") = coefficients.c_str();
      if (bounded) {
        openmc::model::surfaces.push_back(std::make_unique<CountingSphere>(
          sphere, distance_calls[i], sense_calls[i]));
      } else {
        openmc::model::surfaces.push_back(
          std::make_unique<UnboundedCountingSphere>(
            sphere, distance_calls[i], sense_calls[i]));
      }
      openmc::model::surface_map[i + 1] = i;
    }
  }

  ~BoundedBranchFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }

  std::array<int, N_SURFACES> distance_calls {};
  std::array<int, N_SURFACES> sense_calls {};
};

} // anonymous namespace

TEST_CASE("Test region simplification")
{
  SurfaceFixture fixture;

  SECTION("Original bug case from issue #3685")
  {
    // Input: "-1 2 (-3 4) | (-5 6)" was being incorrectly interpreted
    auto region = openmc::Region("(-1 2 (-3 4) | (-5 6))", 0);
    REQUIRE(region.str() == " ( ( -1 2 ( -3 4 ) ) | ( -5 6 ) )");
  }

  SECTION("Simple union - no extra parentheses needed")
  {
    auto region = openmc::Region("1 | 2", 0);
    REQUIRE(region.str() == " 1 | 2");
  }

  SECTION("Intersection then union")
  {
    // Intersection should have higher precedence, so (1 2) grouped
    auto region = openmc::Region("1 2 | 3", 0);
    REQUIRE(region.str() == " ( 1 2 ) | 3");
  }

  SECTION("Union then intersection")
  {
    // The (2 3) intersection should be grouped
    auto region = openmc::Region("1 | 2 3", 0);
    REQUIRE(region.str() == " 1 | ( 2 3 )");
  }

  SECTION("Nested parentheses preserved")
  {
    // These parentheses are meaningful and should be preserved
    auto region = openmc::Region("(1 | 2) (3 | 4)", 0);
    REQUIRE(region.str() == " ( 1 | 2 ) ( 3 | 4 )");
  }

  SECTION("Deep nesting")
  {
    auto region = openmc::Region("((1 2) | (3 4)) 5", 0);
    REQUIRE(region.str() == " ( ( 1 2 ) | ( 3 4 ) ) 5");
  }

  SECTION("Multiple unions")
  {
    auto region = openmc::Region("1 | 2 | 3", 0);
    REQUIRE(region.str() == " 1 | 2 | 3");
  }

  SECTION("Multiple intersections")
  {
    auto region = openmc::Region("1 2 3", 0);
    // Simple cell - no operators in output
    REQUIRE(region.str() == " 1 2 3");
  }

  SECTION("Complex mixed expression")
  {
    auto region = openmc::Region("1 2 | 3 4 | 5 6", 0);
    REQUIRE(region.str() == " ( 1 2 ) | ( 3 4 ) | ( 5 6 )");
  }
}

TEST_CASE("Find boundary after virtual surface crossings")
{
  MultiIntersectionFixture fixture;
  openmc::Region region("-1 | -2", 0);

  SECTION("Starting inside the region")
  {
    // Along +x from x=1, entering the sphere at x=4 is virtual because x < 5.
    // Crossing the plane at x=5 is also virtual because the point is inside
    // the sphere. Exiting the sphere at x=6 is the first true boundary.
    auto [distance, surface] =
      region.distance({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0);

    REQUIRE(distance == Catch::Approx(5.0));
    REQUIRE(surface == 2);
  }

  SECTION("Starting outside the region")
  {
    // Along -x from x=7, entering the sphere at x=6 is the first boundary.
    auto [distance, surface] =
      region.distance({7.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 0);

    REQUIRE(distance == Catch::Approx(1.0));
    REQUIRE(surface == -2);
  }

  SECTION("Starting on a curved surface")
  {
    // Start on the sphere and travel obliquely through it. The plane crossing
    // is virtual, and accumulated roundoff must not cause the sphere exit to
    // be classified as another virtual crossing.
    auto [distance, surface] =
      region.distance({4.2, 0.6, 0.0}, {1.0, 0.0, 0.0}, -2);

    REQUIRE(distance == Catch::Approx(1.6));
    REQUIRE(surface == 2);
  }
}

TEST_CASE("Ignore roundoff-scale virtual surface crossings")
{
  CoincidentSurfaceFixture fixture;

  // These two surfaces describe effectively coincident cylinders, but the
  // small rotation in the general quadric causes their calculated
  // intersections to differ by roundoff. The nearby quadric intersection is
  // virtual and the next meaningful crossing is on the far side of the
  // cylinders.
  openmc::Region region("1 | -2", 0);
  auto [distance, surface] = region.distance(
    {-427.64056354508085, -257.1469395319449, -20.851278766740666},
    {0.8131471271523302, -0.39377148555937275, 0.4286441026822566}, -1);

  REQUIRE(distance == Catch::Approx(603.9161175466262));
  REQUIRE(surface == 1);
}

TEST_CASE("Predecode unique region surfaces")
{
  CountingPlaneFixture fixture;
  openmc::Region region("-1 | -1", 0);

  auto [distance, surface] =
    region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true);

  REQUIRE(distance == Catch::Approx(1.0));
  REQUIRE(surface == 1);
  REQUIRE(fixture.distance_calls[0] == 0);
}

TEST_CASE("Plane-only complex distance precomputes roots")
{
  CountingPlaneFixture fixture;
  openmc::Region region("-1 | -2", 0);

  auto [distance, surface] =
    region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true);

  REQUIRE(distance == Catch::Approx(2.0));
  REQUIRE(surface == 2);
  REQUIRE(fixture.distance_calls[0] == 0);
  REQUIRE(fixture.distance_calls[1] == 0);
  REQUIRE(fixture.sense_calls[0] + fixture.sense_calls[1] == 0);
}

TEST_CASE("Known-inside complex distance skips initial membership test")
{
  CountingPlaneFixture fixture;
  openmc::Region region("-1 | -2", 0);

  const auto known_inside =
    region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true);
  fixture.sense_calls = {};

  const auto default_result =
    region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0);

  REQUIRE(default_result == known_inside);
  REQUIRE(fixture.sense_calls[0] + fixture.sense_calls[1] == 0);
}

TEST_CASE("Complex contains short circuits parenthesized branches")
{
  CountingPlaneFixture fixture;
  openmc::Region region("(-1 | -2) -3", 0);

  REQUIRE(region.contains({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0));
  REQUIRE(fixture.sense_calls[0] == 1);
  REQUIRE(fixture.sense_calls[1] == 0);
  REQUIRE(fixture.sense_calls[2] == 1);
}

TEST_CASE("Complex distance honors an upper bound")
{
  CountingPlaneFixture fixture;
  openmc::Region region("-1 | -2", 0);

  auto [distance, surface] = region.distance(
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true, 0.5);

  REQUIRE(distance == openmc::INFTY);
  REQUIRE(surface == std::numeric_limits<int32_t>::max());
}

TEST_CASE("Initial nearby plane crossing is not suppressed")
{
  ScaledPlaneFixture fixture;
  openmc::Region region("-1 | -1", 0);

  auto [distance, surface] =
    region.distance({-5.0e-13, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0);

  REQUIRE(distance == Catch::Approx(5.0e-13));
  REQUIRE(distance < openmc::FP_COINCIDENT);
  REQUIRE(surface == 1);
}

TEST_CASE("Predecoded plane kernels match virtual surface methods")
{
  auto check = [](const char* type, const char* coefficients,
                 openmc::Position r, openmc::Direction u,
                 int32_t on_surface = 0) {
    PlaneKernelFixture fixture(type, coefficients);
    openmc::Region region("-1 | -1", 0);
    const auto& plane = *openmc::model::surfaces[0];
    const bool coincident = std::abs(on_surface) == 1;
    const double expected_distance = plane.distance(r, u, coincident);
    int32_t expected_surface = std::numeric_limits<int32_t>::max();
    if (expected_distance != openmc::INFTY) {
      const auto r_hit = r + expected_distance * u;
      expected_surface = u.dot(plane.normal(r_hit)) > 0.0 ? 1 : -1;
    }
    const int distance_calls = fixture.distance_calls;
    const int normal_calls = fixture.normal_calls;
    const int sense_calls = fixture.sense_calls;

    const auto [distance, surface] = region.distance(r, u, on_surface);

    REQUIRE(distance == expected_distance);
    REQUIRE(surface == expected_surface);
    REQUIRE(fixture.distance_calls == distance_calls);
    REQUIRE(fixture.normal_calls == normal_calls);
    REQUIRE(fixture.sense_calls == sense_calls);
  };

  SECTION("X plane")
  {
    check("x-plane", "1", {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
  }
  SECTION("Y plane")
  {
    check("y-plane", "2", {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
  }
  SECTION("Z plane")
  {
    check("z-plane", "-1", {0.0, 0.0, 0.0}, {0.0, -1.0, 0.0});
  }
  SECTION("General plane")
  {
    check("plane", "1 2 3 4", {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
  }
  SECTION("Coincident plane")
  {
    check("x-plane", "1", {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, -1);
  }
  SECTION("Near-parallel general plane")
  {
    check("plane", "1 0 0 0", {-1.0e-3, 0.0, 0.0},
      {1.0e-12, 1.0, 0.0});
  }
}

TEST_CASE("Curved complex distance lazily requests successor roots")
{
  CurvedEventFixture fixture;
  openmc::Region region("-2 | -1", 0);

  auto [distance, surface] =
    region.distance({1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0);

  REQUIRE(distance == Catch::Approx(5.0));
  REQUIRE(surface == 1);
  REQUIRE(fixture.distance_calls[0] == 2);
  REQUIRE(fixture.distance_calls[1] == 1);
  REQUIRE(fixture.sense_calls[0] == 0);
  REQUIRE(fixture.sense_calls[1] == 0);
}

TEST_CASE("Tangent curved event preserves boundary semantics")
{
  CurvedEventFixture fixture;
  openmc::Region region("-1 | -1", 0);

  auto [distance, surface] =
    region.distance({3.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, 0);

  REQUIRE(distance == Catch::Approx(2.0));
  REQUIRE(surface == -1);
  REQUIRE(fixture.distance_calls[0] == 2);
}

TEST_CASE("Repeated curved surfaces share distance and sense caches")
{
  CurvedEventFixture fixture;
  openmc::Region region("-1 | -1", 0);

  auto [distance, surface] =
    region.distance({3.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0);

  REQUIRE(distance == Catch::Approx(1.0));
  REQUIRE(surface == -1);
  REQUIRE(fixture.distance_calls[0] == 1);
  REQUIRE(fixture.sense_calls[0] == 1);
}

TEST_CASE("Coincident curved event groups use exact fallback")
{
  CurvedEventFixture fixture(true);
  openmc::Region region("-1 | -2", 0);

  auto [distance, surface] =
    region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0);

  REQUIRE(distance == Catch::Approx(1.0));
  REQUIRE(surface == 1);
  REQUIRE(fixture.distance_calls[0] == 2);
  REQUIRE(fixture.distance_calls[1] == 2);
}

TEST_CASE("Compiled bounds prune disjoint distance branches")
{
  BoundedBranchFixture fixture;
  openmc::Region region(
    "-1 | -2 | -3 | -4 | -5 | -6 | -7 | -8 | -9 | -10 | -11 | -12", 0);

  SECTION("Containment retains canonical evaluation order")
  {
    REQUIRE(region.contains({0.0, 44.0, 0.0}, {1.0, 0.0, 0.0}, 0));
    REQUIRE(std::accumulate(
              fixture.sense_calls.begin(), fixture.sense_calls.end(), 0) == 12);
    REQUIRE(fixture.sense_calls.back() == 1);
  }

  SECTION("Ray intervals avoid irrelevant surface roots")
  {
    const auto [distance, surface] =
      region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true);
    REQUIRE(distance == Catch::Approx(1.0));
    REQUIRE(surface == 1);
    REQUIRE(std::accumulate(fixture.distance_calls.begin(),
              fixture.distance_calls.end(), 0) == 1);
    REQUIRE(fixture.distance_calls.front() == 1);
  }

  SECTION("Ray intervals honor the caller distance limit")
  {
    const auto [distance, surface] =
      region.distance({-2.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, false, 0.5);
    REQUIRE(distance == openmc::INFTY);
    REQUIRE(surface == std::numeric_limits<int32_t>::max());
    REQUIRE(std::accumulate(fixture.distance_calls.begin(),
              fixture.distance_calls.end(), 0) == 0);
  }
}

TEST_CASE("Intersection bounds remove impossible branch work")
{
  BoundedBranchFixture fixture;
  openmc::Region region("(-2 -3) | -1", 0);

  REQUIRE(region.contains({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0));
  REQUIRE(fixture.sense_calls[1] == 1);
  REQUIRE(fixture.sense_calls[2] == 0);
  fixture.sense_calls = {};
  auto [distance, surface] =
    region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true);

  REQUIRE(distance == Catch::Approx(1.0));
  REQUIRE(surface == 1);
  REQUIRE(fixture.distance_calls[1] == 0);
  REQUIRE(fixture.distance_calls[2] == 0);
}

TEST_CASE("Unbounded branches retain exact surface work")
{
  BoundedBranchFixture fixture;
  openmc::Region region("1 | -2", 0);

  const auto [distance, surface] =
    region.distance({2.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true);

  REQUIRE(distance == openmc::INFTY);
  REQUIRE(surface == std::numeric_limits<int32_t>::max());
  REQUIRE(fixture.distance_calls[0] == 1);
}

TEST_CASE("Canonical containment preserves authoritative surface sense")
{
  SurfaceFixture fixture;
  openmc::Region region("(1 -2) | -3", 0);

  REQUIRE(region.contains({-10.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1));
}

TEST_CASE("Bounded candidates match canonical full-surface traversal")
{
  auto run = [](bool bounded) {
    BoundedBranchFixture fixture(bounded);
    openmc::Region region(
      "-1 | -2 | -3 | -4 | -5 | -6 | -7 | -8 | -9 | -10 | -11 | -12", 0);
    const auto result =
      region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true);
    const int calls = std::accumulate(
      fixture.distance_calls.begin(), fixture.distance_calls.end(), 0);
    return std::pair {result, calls};
  };

  const auto bounded = run(true);
  const auto canonical = run(false);
  REQUIRE(bounded.first == canonical.first);
  REQUIRE(bounded.second == 1);
  REQUIRE(canonical.second == BoundedBranchFixture::N_SURFACES);
}

TEST_CASE("Missed bounds fall back to canonical surface traversal")
{
  struct Cleanup {
    ~Cleanup()
    {
      openmc::model::surfaces.clear();
      openmc::model::surface_map.clear();
    }
  } cleanup;
  int distance_calls {0};
  int sense_calls {0};
  pugi::xml_document doc;
  auto sphere = doc.append_child("surface");
  sphere.append_attribute("id") = 1;
  sphere.append_attribute("type") = "sphere";
  sphere.append_attribute("coeffs") = "0 0 0 1";
  openmc::model::surfaces.push_back(
    std::make_unique<MisboundedCountingSphere>(
      sphere, distance_calls, sense_calls));
  openmc::model::surface_map[1] = 0;

  {
    openmc::Region region("-1 | -1", 0);
    const auto [distance, surface] =
      region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true);
    REQUIRE(distance == Catch::Approx(1.0));
    REQUIRE(surface == 1);
    REQUIRE(distance_calls == 1);
  }

}

TEST_CASE("Compiled Boolean constants preserve canonical expressions")
{
  BoundedBranchFixture fixture;

  SECTION("Complementary union is always true")
  {
    openmc::Region region("1 | -1", 0);
    REQUIRE(region.contains({100.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0));
    const auto [distance, surface] =
      region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0, true);
    REQUIRE(distance == openmc::INFTY);
    REQUIRE(surface == std::numeric_limits<int32_t>::max());
    REQUIRE(std::accumulate(fixture.distance_calls.begin(),
              fixture.distance_calls.end(), 0) == 0);
  }

  SECTION("Union of contradictory branches is always false")
  {
    openmc::Region region("(1 -1) | (2 -2)", 0);
    REQUIRE_FALSE(region.contains({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0));
    const auto [distance, surface] =
      region.distance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0);
    REQUIRE(distance == openmc::INFTY);
    REQUIRE(surface == std::numeric_limits<int32_t>::max());
    REQUIRE(std::accumulate(fixture.distance_calls.begin(),
              fixture.distance_calls.end(), 0) == 0);
  }
}
