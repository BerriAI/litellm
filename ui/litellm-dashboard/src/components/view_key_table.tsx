'use client';
import { StatusOnlineIcon } from "@heroicons/react/outline";
import {
  Badge,
  Card,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Text,
  Title,
} from "@tremor/react";

const data = [
  {
    key_alias: "my test key",
    key_name: "sk-...hd74",
    spend: 23.0,
    expires: "active",
    token: "23902dwojd90"
  },
  {
    key_alias: "my test key",
    key_name: "sk-...hd74",
    spend: 23.0,
    expires: "active",
    token: "23902dwojd90"
  },
  {
    key_alias: "my test key",
    key_name: "sk-...hd74",
    spend: 23.0,
    expires: "active",
    token: "23902dwojd90"
  },
  {
    key_alias: "my test key",
    key_name: "sk-...hd74",
    spend: 23.0,
    expires: "active",
    token: "23902dwojd90"
  },
];

export default function ViewKeyTable() { 
    
    return (
  <Card className="flex-auto overflow-y-auto max-h-[50vh] mb-4">
    <Title>API Keys</Title>
    <Table className="mt-5">
      <TableHead>
        <TableRow>
          <TableHeaderCell>Alias</TableHeaderCell>
          <TableHeaderCell>Secret Key</TableHeaderCell>
          <TableHeaderCell>Spend</TableHeaderCell>
          <TableHeaderCell>Status</TableHeaderCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {data.map((item) => (
          <TableRow key={item.token}>
            <TableCell>{item.key_alias}</TableCell>
            <TableCell>
              <Text>{item.key_name}</Text>
            </TableCell>
            <TableCell>
              <Text>{item.spend}</Text>
            </TableCell>
            <TableCell>
              <Badge color="emerald" icon={StatusOnlineIcon}>
                {item.expires}
              </Badge>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </Card>
)};