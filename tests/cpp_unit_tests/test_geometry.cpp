#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openmc/cell.h"
#include "openmc/capi.h"
#include "openmc/geometry.h"
#include "openmc/geometry_aux.h"
#include "openmc/lattice.h"
#include "openmc/model/geometry/compiled_geometry.h"
#include "openmc/particle_data.h"
#include "openmc/surface.h"
#include "openmc/universe.h"

#include <memory>
#include <string>
#include <utility>

#include <pugixml.hpp>

namespace {

class GeometryFixture {
public:
  GeometryFixture()
    : root_universe_ {openmc::model::root_universe},
      n_coord_levels_ {openmc::model::n_coord_levels}
  {
    openmc::model::geometry::clear();
    openmc::model::cells.clear();
    openmc::model::cell_map.clear();
    openmc::model::universes.clear();
    openmc::model::universe_map.clear();
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();

    openmc::model::n_coord_levels = 2;
    openmc::model::root_universe = 0;

    pugi::xml_document surface_doc;
    auto surface_node = surface_doc.append_child("surface");
    surface_node.append_attribute("id") = 1;
    surface_node.append_attribute("type") = "sphere";
    surface_node.append_attribute("coeffs") = "0 0 0 1";
    openmc::model::surfaces.push_back(
      std::make_unique<openmc::SurfaceSphere>(surface_node));
    openmc::model::surface_map[1] = 0;

    openmc::model::cells.push_back(make_cell(1, 0, 1, ""));
    openmc::model::cells.push_back(make_cell(2, 1, -1, "-1"));
    openmc::model::cells.push_back(make_cell(3, 1, -1, "+1"));
    for (int i = 0; i < openmc::model::cells.size(); ++i)
      openmc::model::cell_map[openmc::model::cells[i]->id_] = i;

    auto root = std::make_unique<openmc::Universe>();
    root->id_ = 0;
    root->cells_ = {0};
    root->n_instances_ = 1;
    openmc::model::universes.push_back(std::move(root));
    openmc::model::universe_map[0] = 0;

    auto nested = std::make_unique<openmc::Universe>();
    nested->id_ = 1;
    nested->cells_ = {1, 2};
    nested->n_instances_ = 1;
    openmc::model::universes.push_back(std::move(nested));
    openmc::model::universe_map[1] = 1;
  }

  ~GeometryFixture()
  {
    openmc::model::geometry::clear();
    openmc::model::cells.clear();
    openmc::model::cell_map.clear();
    openmc::model::universes.clear();
    openmc::model::universe_map.clear();
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
    openmc::model::root_universe = root_universe_;
    openmc::model::n_coord_levels = n_coord_levels_;
  }

private:
  static std::unique_ptr<openmc::CSGCell> make_cell(
    int id, int universe, int fill, const char* region)
  {
    pugi::xml_document doc;
    auto node = doc.append_child("cell");
    node.append_attribute("id") = id;
    node.append_attribute("universe") = universe;
    if (fill >= 0) {
      const auto fill_value {std::to_string(fill)};
      node.append_child("fill").text() = fill_value.c_str();
    } else {
      node.append_child("material").text() = "void";
    }
    if (region[0] != '\0')
      node.append_child("region").text() = region;

    auto cell = std::make_unique<openmc::CSGCell>(node);
    if (fill < 0) {
      cell->type_ = openmc::Fill::MATERIAL;
      cell->sqrtkT_.push_back(0.0);
      cell->density_mult_.push_back(1.0);
    } else {
      cell->type_ = openmc::Fill::UNIVERSE;
    }
    return cell;
  }

  int root_universe_;
  int n_coord_levels_;
};

class CountingXPlane : public openmc::SurfaceXPlane {
public:
  CountingXPlane(pugi::xml_node node, int& normal_calls)
    : SurfaceXPlane {node}, normal_calls_ {normal_calls}
  {}

  openmc::Direction normal(openmc::Position r) const override
  {
    ++normal_calls_;
    return SurfaceXPlane::normal(r);
  }

private:
  int& normal_calls_;
};

class BoundaryFixture {
public:
  BoundaryFixture()
    : root_universe_ {openmc::model::root_universe},
      n_coord_levels_ {openmc::model::n_coord_levels}
  {
    openmc::model::cells.clear();
    openmc::model::cell_map.clear();
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
    openmc::model::n_coord_levels = 1;
    openmc::model::root_universe = 0;

    pugi::xml_document surface_doc;
    auto surface_node = surface_doc.append_child("surface");
    surface_node.append_attribute("id") = 1;
    surface_node.append_attribute("type") = "x-plane";
    surface_node.append_attribute("coeffs") = 1;
    openmc::model::surfaces.push_back(
      std::make_unique<CountingXPlane>(surface_node, normal_calls));
    openmc::model::surface_map[1] = 0;

    pugi::xml_document cell_doc;
    auto cell_node = cell_doc.append_child("cell");
    cell_node.append_attribute("id") = 1;
    cell_node.append_attribute("universe") = 0;
    cell_node.append_child("material").text() = "void";
    cell_node.append_child("region").text() = "-1 | -1";
    auto cell = std::make_unique<openmc::CSGCell>(cell_node);
    cell->type_ = openmc::Fill::MATERIAL;
    openmc::model::cells.push_back(std::move(cell));
    openmc::model::cell_map[1] = 0;
  }

  ~BoundaryFixture()
  {
    openmc::model::cells.clear();
    openmc::model::cell_map.clear();
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
    openmc::model::root_universe = root_universe_;
    openmc::model::n_coord_levels = n_coord_levels_;
  }

  int normal_calls {0};

private:
  int root_universe_;
  int n_coord_levels_;
};

class BoundedTestCell : public openmc::Cell {
public:
  BoundedTestCell(int id, int universe, openmc::BoundingBox exact,
    openmc::BoundingBox reported, openmc::vector<int>* calls = nullptr)
    : exact_ {exact}, reported_ {reported}, calls_ {calls}
  {
    id_ = id;
    universe_ = universe;
    fill_ = openmc::C_NONE;
    type_ = openmc::Fill::MATERIAL;
    material_ = {openmc::MATERIAL_VOID};
    sqrtkT_ = {0.0};
    density_mult_ = {1.0};
  }

  bool contains(openmc::Position r, openmc::Direction, int32_t) const override
  {
    if (calls_)
      calls_->push_back(id_);
    return r.x >= exact_.min.x && r.x <= exact_.max.x &&
           r.y >= exact_.min.y && r.y <= exact_.max.y &&
           r.z >= exact_.min.z && r.z <= exact_.max.z;
  }

  std::pair<double, int32_t> distance(openmc::Position, openmc::Direction,
    int32_t, openmc::GeometryState*, bool, double) const override
  {
    return {openmc::INFTY, openmc::SURFACE_NONE};
  }

  void to_hdf5_inner(hid_t) const override {}
  openmc::BoundingBox bounding_box() const override { return reported_; }
  openmc::GeometryType geom_type() const override
  {
    return openmc::GeometryType::CSG;
  }

private:
  openmc::BoundingBox exact_;
  openmc::BoundingBox reported_;
  openmc::vector<int>* calls_;
};

class LimitedDistanceCell : public openmc::Cell {
public:
  LimitedDistanceCell(int id, double boundary, openmc::vector<int>& calls,
    openmc::vector<double>& limits, std::size_t cost = 1)
    : boundary_ {boundary}, cost_ {cost}, calls_ {calls}, limits_ {limits}
  {
    id_ = id;
    fill_ = openmc::C_NONE;
    type_ = openmc::Fill::MATERIAL;
  }

  bool contains(openmc::Position, openmc::Direction, int32_t) const override
  {
    return true;
  }

  std::pair<double, int32_t> distance(openmc::Position, openmc::Direction,
    int32_t, openmc::GeometryState*, bool, double max_distance) const override
  {
    calls_.push_back(id_);
    limits_.push_back(max_distance);
    return boundary_ < max_distance
             ? std::pair<double, int32_t> {boundary_, id_}
             : std::pair<double, int32_t> {
                 openmc::INFTY, openmc::SURFACE_NONE};
  }

  void to_hdf5_inner(hid_t) const override {}
  openmc::BoundingBox bounding_box() const override
  {
    return openmc::BoundingBox::infinite();
  }
  openmc::GeometryType geom_type() const override
  {
    return openmc::GeometryType::CSG;
  }
  std::size_t boundary_search_cost() const override { return cost_; }

private:
  double boundary_;
  std::size_t cost_;
  openmc::vector<int>& calls_;
  openmc::vector<double>& limits_;
};

class AcceleratorFixture {
public:
  AcceleratorFixture()
    : root_universe_ {openmc::model::root_universe},
      n_coord_levels_ {openmc::model::n_coord_levels}
  {
    openmc::model::cells.clear();
    openmc::model::cell_map.clear();
    openmc::model::universes.clear();
    openmc::model::universe_map.clear();
    openmc::model::n_coord_levels = 1;
    openmc::model::root_universe = 0;
  }

  ~AcceleratorFixture()
  {
    openmc::model::cells.clear();
    openmc::model::cell_map.clear();
    openmc::model::universes.clear();
    openmc::model::universe_map.clear();
    openmc::model::root_universe = root_universe_;
    openmc::model::n_coord_levels = n_coord_levels_;
  }

  int add_cell(int universe, openmc::BoundingBox exact,
    openmc::BoundingBox reported, openmc::vector<int>* calls = nullptr)
  {
    const int index = openmc::model::cells.size();
    const int id = 100 + index;
    openmc::model::cells.push_back(std::make_unique<BoundedTestCell>(
      id, universe, exact, reported, calls));
    openmc::model::cell_map[id] = index;
    return index;
  }

  openmc::Universe& add_universe(int id, openmc::vector<int32_t> cells)
  {
    auto universe = std::make_unique<openmc::Universe>();
    universe->id_ = id;
    universe->cells_ = std::move(cells);
    universe->n_instances_ = 1;
    const int index = openmc::model::universes.size();
    openmc::model::universes.push_back(std::move(universe));
    openmc::model::universe_map[id] = index;
    return *openmc::model::universes.back();
  }

private:
  int root_universe_;
  int n_coord_levels_;
};

openmc::BoundingBox box(double xmin, double xmax)
{
  return {{xmin, -1.0, -1.0}, {xmax, 1.0, 1.0}};
}

} // namespace

TEST_CASE("Reconcile a particle after a collision near a surface")
{
  GeometryFixture fixture;

  openmc::GeometryState p;
  p.n_coord() = 2;
  p.coord(0).universe() = 0;
  p.coord(0).cell() = 0;
  p.coord(0).r() = {0.0, 0.0, 0.0};
  p.coord(0).u() = {-1.0, 0.0, 0.0};
  p.coord(1).universe() = 1;
  p.coord(1).cell() = 2;
  p.coord(1).r() = {1.0 - 1.0e-13, 0.0, 0.0};
  p.coord(1).u() = {-1.0, 0.0, 0.0};

  openmc::reconcile_cell_after_collision(p);
  REQUIRE(p.n_coord() == 2);
  REQUIRE(p.coord(0).cell() == 0);
  REQUIRE(p.coord(1).cell() == 1);
}

TEST_CASE("Distance to boundary uses the predecoded plane projection")
{
  BoundaryFixture fixture;
  openmc::GeometryState p;
  p.coord(0).cell() = 0;
  p.coord(0).r() = {0.0, 0.0, 0.0};
  p.coord(0).u() = {1.0, 0.0, 0.0};

  const auto boundary = openmc::distance_to_boundary(p);

  REQUIRE(boundary.distance() == 1.0);
  REQUIRE(boundary.surface() == 1);
  REQUIRE(fixture.normal_calls == 0);
}

TEST_CASE("Inner boundaries clip enclosing region searches")
{
  const int old_levels = openmc::model::n_coord_levels;
  openmc::model::n_coord_levels = 2;
  openmc::vector<int> calls;
  openmc::vector<double> limits;
  openmc::model::cells.clear();
  openmc::model::cells.push_back(
    std::make_unique<LimitedDistanceCell>(10, 10.0, calls, limits, 100));
  openmc::model::cells.push_back(
    std::make_unique<LimitedDistanceCell>(20, 1.0, calls, limits));

  openmc::GeometryState p;
  p.n_coord() = 2;
  p.coord(0).cell() = 0;
  p.coord(1).cell() = 1;
  const auto boundary = openmc::distance_to_boundary(p);

  REQUIRE(boundary.distance() == 1.0);
  REQUIRE(boundary.coord_level() == 2);
  REQUIRE(calls == openmc::vector<int> {20, 10});
  REQUIRE(limits[0] == openmc::INFTY);
  REQUIRE(limits[1] > 1.0);
  REQUIRE(limits[1] < 1.001);

  openmc::model::cells.clear();
  openmc::model::n_coord_levels = old_levels;
}

TEST_CASE("Cheap enclosing boundaries clip expensive inner searches")
{
  const int old_levels = openmc::model::n_coord_levels;
  openmc::model::n_coord_levels = 2;
  openmc::vector<int> calls;
  openmc::vector<double> limits;
  openmc::model::cells.clear();
  openmc::model::cells.push_back(
    std::make_unique<LimitedDistanceCell>(10, 1.0, calls, limits));
  openmc::model::cells.push_back(
    std::make_unique<LimitedDistanceCell>(20, 10.0, calls, limits, 100));

  openmc::GeometryState p;
  p.n_coord() = 2;
  p.coord(0).cell() = 0;
  p.coord(1).cell() = 1;
  const auto boundary = openmc::distance_to_boundary(p);

  REQUIRE(boundary.distance() == 1.0);
  REQUIRE(boundary.coord_level() == 1);
  REQUIRE(calls == openmc::vector<int> {10, 20});
  REQUIRE(limits[0] == openmc::INFTY);
  REQUIRE(limits[1] > 1.0);
  REQUIRE(limits[1] < 1.001);

  openmc::model::cells.clear();
  openmc::model::n_coord_levels = old_levels;
}

TEST_CASE("Coincident enclosing boundaries retain precedence")
{
  const int old_levels = openmc::model::n_coord_levels;
  openmc::model::n_coord_levels = 2;
  openmc::vector<int> calls;
  openmc::vector<double> limits;
  openmc::model::cells.clear();
  openmc::model::cells.push_back(
    std::make_unique<LimitedDistanceCell>(10, 1.0, calls, limits));
  openmc::model::cells.push_back(
    std::make_unique<LimitedDistanceCell>(20, 1.0, calls, limits));

  openmc::GeometryState p;
  p.n_coord() = 2;
  p.coord(0).cell() = 0;
  p.coord(1).cell() = 1;
  const auto boundary = openmc::distance_to_boundary(p);

  REQUIRE(boundary.distance() == 1.0);
  REQUIRE(boundary.surface() == 10);
  REQUIRE(boundary.coord_level() == 1);

  openmc::model::cells.clear();
  openmc::model::n_coord_levels = old_levels;
}

TEST_CASE("Universe AABB candidates retain original cell order")
{
  AcceleratorFixture fixture;
  openmc::vector<int> calls;
  openmc::vector<int32_t> cells;
  for (int i = 0; i < 6; ++i)
    cells.push_back(fixture.add_cell(0, box(-1.0, 1.0),
      box(-10.0 + i, 10.0 + i), &calls));
  std::reverse(cells.begin(), cells.end());
  auto& universe = fixture.add_universe(0, cells);
  universe.partitioner_ = std::make_unique<openmc::UniversePartitioner>(universe);

  openmc::GeometryState p;
  p.coord(0).universe() = 0;
  p.coord(0).r() = {0.0, 0.0, 0.0};
  p.coord(0).u() = {1.0, 0.0, 0.0};

  REQUIRE(universe.find_cell(p));
  REQUIRE(p.coord(0).cell() == cells.front());
  REQUIRE(calls.front() == openmc::model::cells[cells.front()]->id_);
}

TEST_CASE("Universe AABB candidates retain infinite cells")
{
  AcceleratorFixture fixture;
  const int infinite = fixture.add_cell(
    0, box(-1.0, 1.0), openmc::BoundingBox::infinite());
  const int bounded = fixture.add_cell(0, box(-1.0, 1.0), box(-1.0, 1.0));
  auto& universe = fixture.add_universe(0, {infinite, bounded});
  universe.partitioner_ = std::make_unique<openmc::UniversePartitioner>(universe);

  openmc::GeometryState p;
  p.coord(0).universe() = 0;
  p.coord(0).r() = {0.0, 0.0, 0.0};

  REQUIRE(universe.find_cell(p));
  REQUIRE(p.coord(0).cell() == infinite);
}

TEST_CASE("Universe AABB candidates remain subject to exact containment")
{
  AcceleratorFixture fixture;
  openmc::vector<int> calls;
  const int false_positive =
    fixture.add_cell(0, box(2.0, 3.0), box(-1.0, 3.0), &calls);
  const int containing =
    fixture.add_cell(0, box(-1.0, 1.0), box(-1.0, 1.0), &calls);
  auto& universe = fixture.add_universe(0, {false_positive, containing});
  universe.partitioner_ = std::make_unique<openmc::UniversePartitioner>(universe);

  openmc::GeometryState p;
  p.coord(0).universe() = 0;
  p.coord(0).r() = {0.0, 0.0, 0.0};

  REQUIRE(universe.find_cell(p));
  REQUIRE(p.coord(0).cell() == containing);
  REQUIRE(calls == openmc::vector<int> {100, 101});
}

TEST_CASE("Universe AABB search falls back to exhaustive point location")
{
  AcceleratorFixture fixture;
  const int containing = fixture.add_cell(0, box(-1.0, 1.0), box(2.0, 3.0));
  auto& universe = fixture.add_universe(0, {containing});
  universe.partitioner_ = std::make_unique<openmc::UniversePartitioner>(universe);

  openmc::GeometryState p;
  p.coord(0).universe() = 0;
  p.coord(0).r() = {0.0, 0.0, 0.0};

  REQUIRE(universe.find_cell(p));
  REQUIRE(p.coord(0).cell() == containing);
}

TEST_CASE("Universe AABB search preserves an omitted earlier cell")
{
  AcceleratorFixture fixture;
  const int first = fixture.add_cell(0, box(-1.0, 1.0), box(2.0, 3.0));
  const int later = fixture.add_cell(0, box(-1.0, 1.0), box(-1.0, 1.0));
  auto& universe = fixture.add_universe(0, {first, later});
  universe.partitioner_ = std::make_unique<openmc::UniversePartitioner>(universe);

  openmc::GeometryState p;
  p.coord(0).universe() = 0;
  p.coord(0).r() = {0.0, 0.0, 0.0};

  REQUIRE(universe.find_cell(p));
  REQUIRE(p.coord(0).cell() == first);
}

TEST_CASE("One accelerated universe is reused through transformed fills")
{
  AcceleratorFixture fixture;
  openmc::model::n_coord_levels = 2;

  const int left = fixture.add_cell(0, box(-11.0, -9.0), box(-11.0, -9.0));
  const int right = fixture.add_cell(0, box(9.0, 11.0), box(9.0, 11.0));
  const int nested = fixture.add_cell(1, box(-1.0, 1.0), box(-1.0, 1.0));
  auto& left_cell = *openmc::model::cells[left];
  left_cell.type_ = openmc::Fill::UNIVERSE;
  left_cell.fill_ = 1;
  left_cell.translation_ = {-10.0, 0.0, 0.0};
  auto& right_cell = *openmc::model::cells[right];
  right_cell.type_ = openmc::Fill::UNIVERSE;
  right_cell.fill_ = 1;
  right_cell.translation_ = {10.0, 0.0, 0.0};

  auto& root = fixture.add_universe(0, {left, right});
  auto& inner = fixture.add_universe(1, {nested});
  root.partitioner_ = std::make_unique<openmc::UniversePartitioner>(root);
  inner.partitioner_ = std::make_unique<openmc::UniversePartitioner>(inner);

  for (double x : {-10.0, 10.0}) {
    openmc::GeometryState p;
    p.init_from_r_u({x, 0.0, 0.0}, {1.0, 0.0, 0.0});
    REQUIRE(openmc::exhaustive_find_cell(p));
    REQUIRE(p.n_coord() == 2);
    REQUIRE(p.coord(1).cell() == nested);
    REQUIRE(p.coord(1).r().x == 0.0);
  }
}

TEST_CASE("Compiled geometry uses canonical model indices")
{
  GeometryFixture fixture;

  openmc::model::geometry::rebuild();
  const auto& compiled = *openmc::model::geometry::compiled;
  const auto& mapping = compiled.mapping();

  REQUIRE(openmc::model::geometry::canonical_mapping_is_valid());
  REQUIRE(mapping.n_cells() == openmc::model::cells.size());
  REQUIRE(mapping.n_universes() == openmc::model::universes.size());
  REQUIRE(mapping.n_lattices() == 0);
  REQUIRE(mapping.n_surfaces() == openmc::model::surfaces.size());
  REQUIRE(mapping.cell(2) == openmc::model::cell_map.at(3));
  REQUIRE(mapping.universe(1) == openmc::model::universe_map.at(1));
  REQUIRE(mapping.surface(0) == openmc::model::surface_map.at(1));
  REQUIRE_THROWS_AS(mapping.cell(3), std::out_of_range);
  REQUIRE_THROWS_AS(mapping.lattice(0), std::out_of_range);
}

TEST_CASE("Compiled geometry invalidates and rebuilds by generation")
{
  GeometryFixture fixture;
  openmc::model::geometry::rebuild();
  REQUIRE(openmc::model::geometry::is_current());

  const auto initial = openmc::model::geometry::generations();
  const double translation[3] {1.0, 2.0, 3.0};
  REQUIRE(openmc_cell_set_translation(0, translation) == 0);
  REQUIRE(openmc::model::geometry::generations().spatial ==
          initial.spatial + 1);
  REQUIRE(openmc::model::geometry::generations().topology == initial.topology);
  REQUIRE_FALSE(openmc::model::geometry::is_current());

  openmc::model::geometry::rebuild();
  REQUIRE(openmc::model::geometry::is_current());
  REQUIRE(openmc::model::geometry::compiled->generations() ==
          openmc::model::geometry::generations());

  const int32_t void_material = openmc::MATERIAL_VOID;
  REQUIRE(openmc_cell_set_fill(1, static_cast<int>(openmc::Fill::MATERIAL), 1,
            &void_material) == 0);
  REQUIRE(openmc::model::geometry::generations().topology ==
          initial.topology + 1);
  REQUIRE(openmc::model::geometry::generations().property ==
          initial.property + 1);
  REQUIRE_FALSE(openmc::model::geometry::is_current());

  int32_t new_cell;
  REQUIRE(openmc_extend_cells(1, &new_cell, nullptr) == 0);
  REQUIRE(openmc_cell_set_id(new_cell, 4) == 0);
  REQUIRE(openmc::model::geometry::generations().identity ==
          initial.identity + 2);
  openmc::model::geometry::rebuild();
  REQUIRE(openmc::model::geometry::is_current());
  REQUIRE(openmc::model::geometry::compiled->mapping().cell(new_cell) ==
          new_cell);

  openmc::free_memory_geometry();
  REQUIRE(openmc::model::geometry::compiled == nullptr);
  REQUIRE_FALSE(openmc::model::geometry::is_current());
}

namespace {

class CompiledTraversalFixture {
public:
  CompiledTraversalFixture()
    : root_universe_ {openmc::model::root_universe},
      n_coord_levels_ {openmc::model::n_coord_levels}
  {
    openmc::model::geometry::clear();
    openmc::model::cells.clear();
    openmc::model::cell_map.clear();
    openmc::model::lattices.clear();
    openmc::model::lattice_map.clear();
    openmc::model::universes.clear();
    openmc::model::universe_map.clear();
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
    openmc::model::n_coord_levels = 3;
    openmc::model::root_universe = 0;

    add_plane(101, "x-plane", -0.75);
    add_plane(102, "x-plane", -0.25);
    add_plane(103, "x-plane", 0.25);
    add_plane(104, "x-plane", 0.75);
    add_plane(201, "y-plane", -0.75);
    add_plane(202, "y-plane", 0.75);
    add_plane(301, "z-plane", -0.75);
    add_plane(302, "z-plane", 0.75);

    auto lattice_cell = make_cell(10, 0, 100, "");
    lattice_cell->type_ = openmc::Fill::LATTICE;
    lattice_cell->fill_ = 0;
    lattice_cell->offset_ = {0};
    add_cell(std::move(lattice_cell));

    auto transformed_cell = make_cell(20, 1, 2, "");
    transformed_cell->type_ = openmc::Fill::UNIVERSE;
    transformed_cell->offset_ = {0};
    add_cell(std::move(transformed_cell));

    auto material_cell = make_cell(30, 2, -1,
      "(+101 -102 +201 -202 +301 -302) | "
      "(+103 -104 +201 -202 +301 -302)");
    material_cell->distribcell_index_ = 0;
    add_cell(std::move(material_cell));

    add_universe(0, {0}, 1);
    add_universe(1, {1}, 2);
    add_universe(2, {2}, 2);

    pugi::xml_document lattice_doc;
    auto lattice_node = lattice_doc.append_child("lattice");
    lattice_node.append_child("id").text() = 100;
    lattice_node.append_child("dimension").text() = "2 1";
    lattice_node.append_child("lower_left").text() = "-2 -1";
    lattice_node.append_child("pitch").text() = "2 2";
    lattice_node.append_child("universes").text() = "1 1";
    auto lattice = std::make_unique<openmc::RectLattice>(lattice_node);
    lattice->adjust_indices();
    lattice->allocate_offset_table(1);
    lattice->offset(0, openmc::array<int, 3> {0, 0, 0}) = 0;
    lattice->offset(0, openmc::array<int, 3> {1, 0, 0}) = 1;
    openmc::model::lattice_map[100] = 0;
    openmc::model::lattices.push_back(std::move(lattice));

    for (auto& universe : openmc::model::universes) {
      universe->partitioner_ =
        std::make_unique<openmc::UniversePartitioner>(*universe);
    }
    openmc::model::geometry::rebuild();
  }

  ~CompiledTraversalFixture()
  {
    openmc::model::geometry::clear();
    openmc::model::cells.clear();
    openmc::model::cell_map.clear();
    openmc::model::lattices.clear();
    openmc::model::lattice_map.clear();
    openmc::model::universes.clear();
    openmc::model::universe_map.clear();
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
    openmc::model::root_universe = root_universe_;
    openmc::model::n_coord_levels = n_coord_levels_;
  }

  openmc::GeometryState locate(openmc::Position r, openmc::Direction u) const
  {
    openmc::GeometryState p;
    p.init_from_r_u(r, u);
    REQUIRE(openmc::exhaustive_find_cell(p));
    return p;
  }

private:
  static void add_plane(int id, const char* type, double location)
  {
    pugi::xml_document doc;
    auto node = doc.append_child("surface");
    node.append_attribute("id") = id;
    node.append_attribute("type") = type;
    node.append_attribute("coeffs") = location;
    openmc::model::surface_map[id] = openmc::model::surfaces.size();
    if (type[0] == 'x') {
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceXPlane>(node));
    } else if (type[0] == 'y') {
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceYPlane>(node));
    } else {
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceZPlane>(node));
    }
  }

  static std::unique_ptr<openmc::CSGCell> make_cell(
    int id, int universe, int fill, const char* region)
  {
    pugi::xml_document doc;
    auto node = doc.append_child("cell");
    node.append_child("id").text() = id;
    node.append_child("universe").text() = universe;
    if (fill >= 0) {
      node.append_child("fill").text() = fill;
    } else {
      node.append_child("material").text() = "void";
    }
    if (region[0] != '\0')
      node.append_child("region").text() = region;

    auto cell = std::make_unique<openmc::CSGCell>(node);
    if (fill < 0) {
      cell->type_ = openmc::Fill::MATERIAL;
      cell->sqrtkT_.push_back(0.0);
      cell->density_mult_.push_back(1.0);
    }
    return cell;
  }

  static void add_cell(std::unique_ptr<openmc::CSGCell> cell)
  {
    openmc::model::cell_map[cell->id_] = openmc::model::cells.size();
    openmc::model::cells.push_back(std::move(cell));
  }

  static void add_universe(
    int id, openmc::vector<int32_t> cells, int n_instances)
  {
    auto universe = std::make_unique<openmc::Universe>();
    universe->id_ = id;
    universe->cells_ = std::move(cells);
    universe->n_instances_ = n_instances;
    openmc::model::universe_map[id] = openmc::model::universes.size();
    openmc::model::universes.push_back(std::move(universe));
  }

  int root_universe_;
  int n_coord_levels_;
};

void require_position(openmc::Position actual, openmc::Position expected)
{
  REQUIRE(actual.x == Catch::Approx(expected.x).margin(1.0e-14));
  REQUIRE(actual.y == Catch::Approx(expected.y).margin(1.0e-14));
  REQUIRE(actual.z == Catch::Approx(expected.z).margin(1.0e-14));
}

void require_canonical_stack(const openmc::GeometryState& p, int lattice_x,
  int instance, openmc::Position leaf_r, openmc::Direction leaf_u)
{
  REQUIRE(p.n_coord() == 3);
  REQUIRE(openmc::model::cells[p.coord(0).cell()]->id_ == 10);
  REQUIRE(openmc::model::cells[p.coord(1).cell()]->id_ == 20);
  REQUIRE(openmc::model::cells[p.coord(2).cell()]->id_ == 30);
  REQUIRE(p.coord(0).universe() == 0);
  REQUIRE(p.coord(1).universe() == 1);
  REQUIRE(p.coord(2).universe() == 2);
  REQUIRE(p.coord(1).lattice() == 0);
  REQUIRE(p.coord(1).lattice_index() ==
          openmc::array<int, 3> {lattice_x, 0, 0});
  require_position(p.coord(2).r(), leaf_r);
  require_position(p.coord(2).u(), leaf_u);
  REQUIRE(p.cell_instance() == instance);
  REQUIRE(openmc::cell_instance_at_level(p, 2) == instance);
}

} // namespace

TEST_CASE("Compiled traversal preserves canonical geometry state")
{
  CompiledTraversalFixture fixture;
  REQUIRE(openmc::model::geometry::is_current());

  auto left = fixture.locate({-0.5, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  auto right = fixture.locate({1.5, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  require_canonical_stack(
    left, 0, 0, {0.5, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  require_canonical_stack(
    right, 1, 1, {0.5, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  REQUIRE(left.coord(1).universe() == right.coord(1).universe());
  REQUIRE(left.coord(2).universe() == right.coord(2).universe());

  const auto coordinates_before = left.coord();
  const auto boundary = openmc::distance_to_boundary(left);
  REQUIRE(boundary.distance() == Catch::Approx(0.25));
  REQUIRE(boundary.surface() == -3);
  REQUIRE(openmc::model::surfaces[boundary.surface_index()]->id_ == 103);
  REQUIRE(boundary.coord_level() == 3);
  require_canonical_stack(
    left, 0, 0, {0.5, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  for (int level = 0; level < left.n_coord(); ++level) {
    REQUIRE(left.coord(level).cell() == coordinates_before[level].cell());
    REQUIRE(left.coord(level).universe() ==
            coordinates_before[level].universe());
    REQUIRE(left.coord(level).lattice() ==
            coordinates_before[level].lattice());
    REQUIRE(left.coord(level).lattice_index() ==
            coordinates_before[level].lattice_index());
    require_position(left.coord(level).r(), coordinates_before[level].r());
    require_position(left.coord(level).u(), coordinates_before[level].u());
  }
}

TEST_CASE("Compiled traversal rebuilds after transform invalidation")
{
  CompiledTraversalFixture fixture;
  const double translation[3] {0.25, 0.0, 0.0};
  REQUIRE(openmc_cell_set_translation(1, translation) == 0);
  REQUIRE_FALSE(openmc::model::geometry::is_current());
  openmc::model::geometry::rebuild();

  auto translated = fixture.locate({-0.25, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  require_canonical_stack(
    translated, 0, 0, {0.5, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  auto boundary = openmc::distance_to_boundary(translated);
  REQUIRE(boundary.distance() == Catch::Approx(0.25));
  REQUIRE(boundary.surface() == -3);

  const double rotation[3] {0.0, 0.0, 180.0};
  REQUIRE(openmc_cell_set_rotation(1, rotation, 3) == 0);
  REQUIRE_FALSE(openmc::model::geometry::is_current());
  openmc::model::geometry::rebuild();

  auto rotated = fixture.locate({-0.25, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  require_canonical_stack(
    rotated, 0, 0, {-0.5, 0.0, 0.0}, {1.0, 0.0, 0.0});
  boundary = openmc::distance_to_boundary(rotated);
  REQUIRE(boundary.distance() == Catch::Approx(0.25));
  REQUIRE(boundary.surface() == 2);
  REQUIRE(openmc::model::surfaces[boundary.surface_index()]->id_ == 102);
}
