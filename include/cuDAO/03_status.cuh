#pragma once
#include "02_enum.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // cuDAO Error
    // ──────────────────────────────────────────────────────────────────────────

    class LazyString {
        char* data{nullptr};
        size_t size_{0};

    public:
        LazyString() noexcept = default;

        explicit LazyString(const std::string& s) noexcept {
            data = new(std::nothrow) char[s.size() + 1];
            if (data) {
                std::memcpy(data, s.c_str(), s.size() + 1);
                size_ = s.size();
            }
        }

        explicit LazyString(const char* s) noexcept {
            if (!s) {
                return;
            }
            size_ = std::strlen(s);
            data = new(std::nothrow) char[size_ + 1];
            if (data) {
                std::memcpy(data, s, size_ + 1);
            }
        }

        LazyString(const LazyString& s) = delete;

        LazyString(LazyString&& s) noexcept : data(s.data), size_(s.size_) {
            s.data = nullptr;
            s.size_ = 0;
        }

        LazyString& operator=(const LazyString& other) = delete;

        LazyString& operator=(LazyString&& other) noexcept {
            if (this != &other) {
                delete[] data;
                data = other.data;
                size_ = other.size_;
                other.data = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        [[nodiscard]] size_t size() const {
            return size_;
        }

        [[nodiscard]] const char* c_str() const {
            return data ? data : "";
        }
    };

    struct cuDAOStatus {
        cuDAOError err;
        CUresult cudaResult;
        const char* where;
        LazyString msg;

        explicit cuDAOStatus(const cuDAOError err_, const char* where_ = nullptr,
                             const CUresult cudaResult_ = CUDA_SUCCESS) noexcept : err(err_), cudaResult(cudaResult_),
            where(where_) {
            switch (err) {
            case cuDAOError::Success:
                break;
            case cuDAOError::SlotPoolExhausted:
                msg = LazyString{"No more version slot available. Too many concurrent tracked pointers."};
                break;
            case cuDAOError::InvalidPtr:
                msg = LazyString{"Invalid pointer."};
                break;
            case cuDAOError::ParameterOverflow:
                msg = LazyString{"Too many parameters."};
                break;
            case cuDAOError::CudaDriverError:
                msg = LazyString{"CUDA driver error."};
                break;
            case cuDAOError::InternalError:
                msg = LazyString{"Internal error. This may be caused by insufficient memory or other factors"};
                break;
            case cuDAOError::HostAllocationFailed:
                msg = LazyString{"Host allocation failed."};
                break;
            case cuDAOError::SynchronizeFailed:
                break;
            case cuDAOError::InvalidDeviceFunctionSymbol:
                msg = LazyString{"Invalid device function symbol."};
                break;
            default:
                msg = LazyString{"Unknown error."};
            }
        }

        cuDAOStatus() : cuDAOStatus(cuDAOError::Success) {
        }

        cuDAOStatus(const cuDAOStatus& other) noexcept : cuDAOStatus(other.err, other.where, other.cudaResult) {
        }

        cuDAOStatus(cuDAOStatus&& other) noexcept
            : err(other.err), cudaResult(other.cudaResult),
              where(other.where), msg(std::move(other.msg)) {
        }

        cuDAOStatus& operator=(cuDAOStatus&& other) noexcept {
            if (this == &other) return *this;
            err = other.err;
            cudaResult = other.cudaResult;
            where = other.where;
            msg = std::move(other.msg);
            return *this;
        }
    };

}
