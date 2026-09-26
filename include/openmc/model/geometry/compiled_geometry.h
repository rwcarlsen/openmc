#ifndef OPENMC_MODEL_GEOMETRY_COMPILED_GEOMETRY_H
#define OPENMC_MODEL_GEOMETRY_COMPILED_GEOMETRY_H

#include <cstddef>
#include <cstdint>

#include "openmc/memory.h"

namespace openmc {
namespace model {
namespace geometry {

enum class GeometryChange : uint8_t {
  TOPOLOGY = 1 << 0,
  SPATIAL = 1 << 1,
  PROPERTY = 1 << 2,
  IDENTITY = 1 << 3
};

constexpr GeometryChange operator|(GeometryChange lhs, GeometryChange rhs)
{
  return static_cast<GeometryChange>(
    static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

struct GeometryGenerations {
  uint64_t topology {0};
  uint64_t spatial {0};
  uint64_t property {0};
  uint64_t identity {0};
};

bool operator==(
  const GeometryGenerations& lhs, const GeometryGenerations& rhs);

inline bool operator!=(
  const GeometryGenerations& lhs, const GeometryGenerations& rhs)
{
  return !(lhs == rhs);
}

//! Sizes of the canonical model index spaces used by compiled geometry data.
//!
//! Compiled data uses the corresponding model container index directly. It
//! does not own a separate ID map or copy properties from model objects.
class CanonicalGeometryMapping {
public:
  CanonicalGeometryMapping(std::size_t n_cells, std::size_t n_universes,
    std::size_t n_lattices, std::size_t n_surfaces);

  std::size_t n_cells() const { return n_cells_; }
  std::size_t n_universes() const { return n_universes_; }
  std::size_t n_lattices() const { return n_lattices_; }
  std::size_t n_surfaces() const { return n_surfaces_; }

  int32_t cell(int32_t canonical_index) const;
  int32_t universe(int32_t canonical_index) const;
  int32_t lattice(int32_t canonical_index) const;
  int32_t surface(int32_t canonical_index) const;

private:
  static int32_t checked_index(
    int32_t canonical_index, std::size_t size, const char* kind);

  std::size_t n_cells_;
  std::size_t n_universes_;
  std::size_t n_lattices_;
  std::size_t n_surfaces_;
};

//! Immutable metadata shared by compiled geometry representations.
class CompiledGeometry {
public:
  CompiledGeometry(
    GeometryGenerations generations, CanonicalGeometryMapping mapping);

  const GeometryGenerations& generations() const { return generations_; }
  const CanonicalGeometryMapping& mapping() const { return mapping_; }

private:
  GeometryGenerations generations_;
  CanonicalGeometryMapping mapping_;
};

extern unique_ptr<const CompiledGeometry> compiled;

const GeometryGenerations& generations();
void mark_dirty(GeometryChange changes);
void rebuild();
void clear();
bool is_current();
bool is_spatial_current();

//! Check that authoritative ID maps agree with canonical model indices.
bool canonical_mapping_is_valid();

} // namespace geometry
} // namespace model
} // namespace openmc

#endif // OPENMC_MODEL_GEOMETRY_COMPILED_GEOMETRY_H
