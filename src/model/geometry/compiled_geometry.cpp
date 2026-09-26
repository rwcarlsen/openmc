#include "openmc/model/geometry/compiled_geometry.h"

#include <cassert>
#include <stdexcept>
#include <string>

#include "openmc/cell.h"
#include "openmc/lattice.h"
#include "openmc/surface.h"
#include "openmc/universe.h"

namespace openmc {
namespace model {
namespace geometry {

bool operator==(
  const GeometryGenerations& lhs, const GeometryGenerations& rhs)
{
  return lhs.topology == rhs.topology && lhs.spatial == rhs.spatial &&
         lhs.property == rhs.property && lhs.identity == rhs.identity;
}

CanonicalGeometryMapping::CanonicalGeometryMapping(std::size_t n_cells,
  std::size_t n_universes, std::size_t n_lattices, std::size_t n_surfaces)
  : n_cells_ {n_cells}, n_universes_ {n_universes},
    n_lattices_ {n_lattices}, n_surfaces_ {n_surfaces}
{}

int32_t CanonicalGeometryMapping::checked_index(
  int32_t canonical_index, std::size_t size, const char* kind)
{
  if (canonical_index < 0 || static_cast<std::size_t>(canonical_index) >= size) {
    throw std::out_of_range {
      std::string {kind} + " canonical index is out of bounds"};
  }
  return canonical_index;
}

int32_t CanonicalGeometryMapping::cell(int32_t canonical_index) const
{
  return checked_index(canonical_index, n_cells_, "Cell");
}

int32_t CanonicalGeometryMapping::universe(int32_t canonical_index) const
{
  return checked_index(canonical_index, n_universes_, "Universe");
}

int32_t CanonicalGeometryMapping::lattice(int32_t canonical_index) const
{
  return checked_index(canonical_index, n_lattices_, "Lattice");
}

int32_t CanonicalGeometryMapping::surface(int32_t canonical_index) const
{
  return checked_index(canonical_index, n_surfaces_, "Surface");
}

CompiledGeometry::CompiledGeometry(
  GeometryGenerations generations, CanonicalGeometryMapping mapping)
  : generations_ {generations}, mapping_ {mapping}
{}

unique_ptr<const CompiledGeometry> compiled;

namespace {
GeometryGenerations current_generations;

bool changed(GeometryChange changes, GeometryChange domain)
{
  return (static_cast<uint8_t>(changes) & static_cast<uint8_t>(domain)) != 0;
}

template<class Objects>
bool map_is_canonical(const Objects& objects,
  const std::unordered_map<int32_t, int32_t>& id_map)
{
  if (objects.size() != id_map.size())
    return false;

  for (std::size_t i = 0; i < objects.size(); ++i) {
    auto entry = id_map.find(objects[i]->id_);
    if (entry == id_map.end() || entry->second != i)
      return false;
  }
  return true;
}
} // namespace

const GeometryGenerations& generations()
{
  return current_generations;
}

void mark_dirty(GeometryChange changes)
{
  if (changed(changes, GeometryChange::TOPOLOGY))
    ++current_generations.topology;
  if (changed(changes, GeometryChange::SPATIAL))
    ++current_generations.spatial;
  if (changed(changes, GeometryChange::PROPERTY))
    ++current_generations.property;
  if (changed(changes, GeometryChange::IDENTITY))
    ++current_generations.identity;
}

bool canonical_mapping_is_valid()
{
  return map_is_canonical(cells, cell_map) &&
         map_is_canonical(universes, universe_map) &&
         map_is_canonical(lattices, lattice_map) &&
         map_is_canonical(surfaces, surface_map);
}

void rebuild()
{
  assert(canonical_mapping_is_valid());
  compiled = make_unique<const CompiledGeometry>(current_generations,
    CanonicalGeometryMapping {
      cells.size(), universes.size(), lattices.size(), surfaces.size()});
}

void clear()
{
  compiled.reset();
}

bool is_current()
{
  return compiled && compiled->generations() == current_generations &&
         compiled->mapping().n_cells() == cells.size() &&
         compiled->mapping().n_universes() == universes.size() &&
         compiled->mapping().n_lattices() == lattices.size() &&
         compiled->mapping().n_surfaces() == surfaces.size();
}

bool is_spatial_current()
{
  if (!compiled)
    return false;
  const auto& built = compiled->generations();
  return built.topology == current_generations.topology &&
         built.spatial == current_generations.spatial &&
         built.identity == current_generations.identity &&
         compiled->mapping().n_cells() == cells.size() &&
         compiled->mapping().n_universes() == universes.size() &&
         compiled->mapping().n_lattices() == lattices.size() &&
         compiled->mapping().n_surfaces() == surfaces.size();
}

} // namespace geometry
} // namespace model
} // namespace openmc
