// Copyright (c) 2026 gpe contributors.
//
// Where a backend looks a kernel up.
//
// Two places, in this order: the table the build embedded, then anything
// registered at run time. The build's table first because it is the one gpe
// tests against and the one that cannot go missing, and because a plugin should
// not be able to shadow `display` or `scale` by naming a kernel after them.
#pragma once

#include <string_view>

#include "gpe_kernels.h"

namespace gpe::kernels {

/// The blob called `name`, compiled in or registered, or null.
[[nodiscard]] const Blob* lookup(std::string_view name);

}   // namespace gpe::kernels
