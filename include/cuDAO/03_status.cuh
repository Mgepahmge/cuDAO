#pragma once
#include "02_enum.cuh"

namespace cuDAO {

    // ──────────────────────────────────────────────────────────────────────────
    // cuDAO Error
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * @brief Small move-only string used inside cuDAOStatus.
     *
     * LazyString avoids throwing from error-message construction. If allocation
     * fails, the string becomes empty rather than propagating an exception.
     */
    class LazyString {
        char* data{nullptr};
        size_t size_{0};

    public:
        /**
         * @brief Construct an empty string.
         */
        LazyString() noexcept = default;

        /**
         * @brief Copy text from a std::string without throwing.
         */
        explicit LazyString(const std::string& s) noexcept {
            data = new(std::nothrow) char[s.size() + 1];
            if (data) {
                std::memcpy(data, s.c_str(), s.size() + 1);
                size_ = s.size();
            }
        }

        /**
         * @brief Copy text from a null-terminated string without throwing.
         */
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

        /**
         * @brief Move-construct a LazyString.
         */
        LazyString(LazyString&& s) noexcept : data(s.data), size_(s.size_) {
            s.data = nullptr;
            s.size_ = 0;
        }

        LazyString& operator=(const LazyString& other) = delete;

        /**
         * @brief Move-assign a LazyString.
         */
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

        /**
         * @brief Return the stored string length.
         */
        [[nodiscard]] size_t size() const {
            return size_;
        }

        /**
         * @brief Return a null-terminated C string.
         *
         * @return Stored message text, or an empty string if no allocation was available.
         */
        [[nodiscard]] const char* c_str() const {
            return data ? data : "";
        }
    };

    /**
     * @brief Status object returned by cuDAO public APIs.
     *
     * cuDAOStatus separates cuDAO-level errors from CUDA Driver API errors. If
     * err is cuDAOError::CudaDriverError, cudaResult stores the original CUDA
     * Driver API result. where identifies the cuDAO function that produced the
     * status when available.
     */
    struct cuDAOStatus {
        cuDAOError err; ///< cuDAO-level result code.
        CUresult cudaResult; ///< CUDA Driver API result associated with the error.
        const char* where; ///< Function name that produced the status, when available.
        LazyString msg; ///< Human-readable message for the cuDAO-level error.

        /**
         * @brief Construct a status from a cuDAO error code.
         *
         * @param err_ cuDAO-level error code.
         * @param where_ Optional function name.
         * @param cudaResult_ Optional CUDA Driver API result.
         */
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

        /**
         * @brief Construct a successful status.
         */
        cuDAOStatus() : cuDAOStatus(cuDAOError::Success) {
        }

        /**
         * @brief Copy status metadata and reconstruct the message from the error code.
         */
        cuDAOStatus(const cuDAOStatus& other) noexcept : cuDAOStatus(other.err, other.where, other.cudaResult) {
        }

        /**
         * @brief Move-construct a status object.
         */
        cuDAOStatus(cuDAOStatus&& other) noexcept
            : err(other.err), cudaResult(other.cudaResult),
              where(other.where), msg(std::move(other.msg)) {
        }

        /**
         * @brief Move-assign a status object.
         */
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
