package litellm

import (
	"fmt"
	"log"
	"strings"
)

// orgMembership is one row of the members list returned by GET /organization/info.
type orgMembership struct {
	UserID    string
	UserEmail string
	Role      string
}

// fetchOrganizationMembers returns the current members of an organization.
// found is false when the proxy reports that the organization does not exist.
func fetchOrganizationMembers(client *Client, organizationID string) (members []orgMembership, found bool, err error) {
	resp, err := client.GetOrganizationInfo(organizationID)
	if err != nil {
		if isNotFound(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("error reading organization %s: %w", organizationID, err)
	}

	rawMembers, _ := resp["members"].([]interface{})
	for _, raw := range rawMembers {
		row, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		member := orgMembership{}
		member.UserID, _ = row["user_id"].(string)
		member.Role, _ = row["user_role"].(string)
		if user, ok := row["user"].(map[string]interface{}); ok {
			member.UserEmail, _ = user["user_email"].(string)
		}
		members = append(members, member)
	}
	return members, true, nil
}

// findOrgMember matches by user_id when one is given, otherwise by user_email
// (case-insensitive). The proxy mints a user_id for members added by email
// only, so email-only configuration has to be matched on the email.
func findOrgMember(members []orgMembership, userID, userEmail string) (orgMembership, bool) {
	for _, member := range members {
		if userID != "" {
			if member.UserID == userID {
				return member, true
			}
			continue
		}
		if userEmail != "" && strings.EqualFold(member.UserEmail, userEmail) {
			return member, true
		}
	}
	return orgMembership{}, false
}

// orgMemberIdentity returns how to address a configured member on the proxy.
// An email-only member is addressed by the user_id the proxy resolved, because
// the email can change while the user_id and the membership stay the same.
func orgMemberIdentity(member, resolvedUserIDs map[string]interface{}) (userID, userEmail string) {
	userID, _ = member["user_id"].(string)
	userEmail, _ = member["user_email"].(string)
	if userID != "" {
		return userID, userEmail
	}
	if resolved, ok := resolvedUserIDs[strings.ToLower(userEmail)].(string); ok && resolved != "" {
		return resolved, ""
	}
	return "", userEmail
}

// deleteOrganizationMember removes a member and treats a member (or
// organization) that is already gone as deleted. The proxy fails the delete
// in that case, which would otherwise block every destroy until the resource
// is removed from state by hand.
func deleteOrganizationMember(client *Client, organizationID, userID, userEmail string) error {
	deleteData := map[string]interface{}{
		"organization_id": organizationID,
	}
	if userID != "" {
		deleteData["user_id"] = userID
	}
	if userEmail != "" {
		deleteData["user_email"] = userEmail
	}

	log.Printf("[DEBUG] Delete organization member request payload: %+v", deleteData)

	_, deleteErr := client.DeleteOrganizationMember(deleteData)
	if deleteErr == nil {
		return nil
	}

	members, found, err := fetchOrganizationMembers(client, organizationID)
	if err != nil {
		return fmt.Errorf("error deleting organization member: %v", deleteErr)
	}
	if !found {
		log.Printf("[WARN] Organization %s no longer exists, treating member as deleted", organizationID)
		return nil
	}
	if _, stillMember := findOrgMember(members, userID, userEmail); !stillMember {
		log.Printf("[WARN] Member (user_id=%q, user_email=%q) is no longer in organization %s, treating as deleted", userID, userEmail, organizationID)
		return nil
	}
	return fmt.Errorf("error deleting organization member: %v", deleteErr)
}
