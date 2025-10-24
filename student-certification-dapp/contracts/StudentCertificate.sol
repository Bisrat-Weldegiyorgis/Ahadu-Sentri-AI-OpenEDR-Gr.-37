// SPDX-License-Identifier: MIT
pragma solidity ^0.8.18;

/**
 * @title StudentCertificate
 * @notice A minimal, access-controlled certificate registry.
 *         Only the contract deployer (issuer) can issue certificates.
 */
contract StudentCertificate {
    address public immutable issuer;

    struct Certificate {
        address student;
        string name;
        string course;
        uint256 year;
        bool issued;
    }

    mapping(address => Certificate) private certificates;

    event CertificateIssued(address indexed student, string name, string course, uint256 year);

    modifier onlyIssuer() {
        require(msg.sender == issuer, "Not authorized: issuer only");
        _;
    }

    constructor() {
        issuer = msg.sender;
    }

    function issueCertificate(
        address student,
        string memory name,
        string memory course,
        uint256 year
    ) external onlyIssuer {
        require(student != address(0), "Invalid student");
        require(bytes(name).length > 0, "Name required");
        require(bytes(course).length > 0, "Course required");
        require(year >= 1950 && year <= 3000, "Invalid year");

        certificates[student] = Certificate({
            student: student,
            name: name,
            course: course,
            year: year,
            issued: true
        });

        emit CertificateIssued(student, name, course, year);
    }

    function getCertificate(address student) external view returns (Certificate memory) {
        return certificates[student];
    }
}
