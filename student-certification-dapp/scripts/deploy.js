const hre = require('hardhat')

async function main() {
  const StudentCertificate = await hre.ethers.getContractFactory('StudentCertificate')
  const contract = await StudentCertificate.deploy()
  await contract.waitForDeployment()
  const address = await contract.getAddress()
  console.log('StudentCertificate deployed to:', address)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
